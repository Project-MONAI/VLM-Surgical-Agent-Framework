# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from abc import ABC, abstractmethod

import json
import logging
import yaml
import time
import tiktoken
import queue
import threading
from threading import Lock
from typing import Any, List, Sequence
import base64
import tempfile
import os
import requests
from openai import OpenAI

class Agent(ABC):
    """
    Common functionality for every agent (chat, note‑taker, selector, …).
    A single, process‑wide lock guarantees that only one thread talks to the
    model at a time – that is important for vLLM’s streaming endpoint.
    """

    _llm_lock = Lock()

    def __init__(self, settings_path, response_handler, agent_key=None, message_bus=None):
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")

        self.load_settings(settings_path, agent_key=agent_key)
        self.response_handler = response_handler

        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.client = OpenAI(api_key="EMPTY", base_url=self.llm_url)

        self._wait_for_server()

        # Message bus integration
        self.message_bus = message_bus
        self.agent_name = type(self).__name__
        self.message_queue = None
        self._message_listener_thread = None

        if self.message_bus:
            self.message_queue = self.message_bus.register_agent(self.agent_name)
            self._start_message_listener()
            self._logger.info(f"{self.agent_name} registered with message bus")

    def load_settings(self, settings_path, agent_key=None):
        """
        Load YAML config and populate the most frequently accessed attributes.
        """
        with open(settings_path, 'r') as f:
            full_config = yaml.safe_load(f)
        if agent_key and agent_key in full_config:
            self.agent_settings = full_config[agent_key]
        else:
            self.agent_settings = full_config

        # Optional overrides for model_name and llm_url via environment variables and a global YAML
        # Precedence: ENV > agent-specific config > global.yaml > default
        # ENV keys:
        #   model_name: VLLM_MODEL_NAME
        #   llm_url:    VLLM_URL
        # Global YAML path: $VLLM_GLOBAL_CONFIG, else <settings_dir>/global.yaml
        # Defaults: model_name='llama3.2', llm_url='http://localhost:8000/v1'
        global_cfg: dict = {}
        try:
            # Allow custom path via env, otherwise look next to the agent settings file
            env_global_cfg_path = os.environ.get("VLLM_GLOBAL_CONFIG")
            candidate_paths = []
            if env_global_cfg_path:
                candidate_paths.append(env_global_cfg_path)
            candidate_paths.append(os.path.join(os.path.dirname(settings_path), "global.yaml"))
            for cfg_path in candidate_paths:
                if cfg_path and os.path.isfile(cfg_path):
                    with open(cfg_path, 'r') as gf:
                        global_cfg = yaml.safe_load(gf) or {}
                    break
        except Exception as e:
            # Non-fatal: proceed without global config
            self._logger.debug(f"No global config loaded: {e}")
        # Expose the parsed global config for downstream agents (e.g., personnel placeholders)
        try:
            self.global_settings = global_cfg if isinstance(global_cfg, dict) else {}
        except Exception:
            self.global_settings = {}

        env_model_name = os.environ.get("VLLM_MODEL_NAME")
        env_served_model_name = os.environ.get("VLLM_SERVED_MODEL_NAME")
        env_llm_url = os.environ.get("VLLM_URL")

        self.description = self.agent_settings.get('description', '')
        self.max_prompt_tokens = self.agent_settings.get('max_prompt_tokens', 3000)
        self.ctx_length = self.agent_settings.get('ctx_length', 2048)
        self.agent_prompt = self.agent_settings.get('agent_prompt', '').strip()
        self.user_prefix = self.agent_settings.get('user_prefix', '')
        self.bot_prefix = self.agent_settings.get('bot_prefix', '')
        self.bot_rule_prefix = self.agent_settings.get('bot_rule_prefix', '')
        self.end_token = self.agent_settings.get('end_token', '')
        self.grammar = self.agent_settings.get('grammar', None)

        self._logger.debug(
            f"Agent config ENV VARS. model_name={env_model_name}"
        )
        # Determine model name; can be an identifier or a path. If it's a relative path,
        # normalize to an absolute path so it matches vLLM's served model name.
        # Prefer a served model name if provided (client/server must match id)
        served_model_name = (
            env_served_model_name
            or self.agent_settings.get('served_model_name')
            or (global_cfg.get('served_model_name') if isinstance(global_cfg, dict) else None)
        )

        # Keep a hint of the configured (path-like) model name to detect families (e.g., Qwen‑VL)
        self.model_name_hint = (
            env_model_name
            or self.agent_settings.get('model_name')
            or (global_cfg.get('model_name') if isinstance(global_cfg, dict) else None)
            or ""
        )

        if served_model_name:
            self.model_name = served_model_name
        else:
            raw_model_name = (
                env_model_name
                or self.agent_settings.get('model_name')
                or (global_cfg.get('model_name') if isinstance(global_cfg, dict) else None)
                or 'llama3.2'
            )
            self.model_name = raw_model_name
            try:
                if isinstance(raw_model_name, str):
                    # Heuristic: looks like a path if it contains a path separator
                    if ("/" in raw_model_name or "\\" in raw_model_name):
                        # If not absolute, make it absolute relative to repo root
                        if not os.path.isabs(raw_model_name):
                            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                            abs_model = os.path.normpath(os.path.join(repo_root, raw_model_name))
                            self.model_name = abs_model
            except Exception:
                # Non-fatal; keep raw string
                self.model_name = raw_model_name
        self.publish_settings = self.agent_settings.get('publish', {})
        self.llm_url = (
            env_llm_url
            or self.agent_settings.get('llm_url')
            or (global_cfg.get('llm_url') if isinstance(global_cfg, dict) else None)
            or "http://localhost:8000/v1"
        )
        self.tools = self.agent_settings.get('tools', {})
        self._logger.debug(
            f"Agent config loaded. llm_url={self.llm_url}, model_name={self.model_name}"
        )

    def _is_qwen_vl(self) -> bool:
        try:
            name = (self.model_name_hint or self.model_name or "").lower()
        except Exception:
            name = str(self.model_name).lower()
        return ("qwen" in name) and ("vl" in name)

    def _wait_for_server(self, timeout=60):
        attempts = 0
        check_url = f"{self.llm_url}/models"
        while attempts < timeout:
            try:
                r = requests.get(check_url)
                if r.status_code == 200:
                    self._logger.info(f"✅ Successfully connected to vLLM server at {self.llm_url}")
                    return
            except Exception as e:
                if attempts % 5 == 0:  # Log less frequently to reduce clutter
                    self._logger.info(f"Waiting for vLLM server (attempt {attempts+1}/{timeout}): {e}")
                else:
                    self._logger.debug(f"Waiting for vLLM server (attempt {attempts+1}/{timeout}): {e}")
            time.sleep(1)
            attempts += 1

        # More helpful error message
        raise ConnectionError(
            f"⚠️ Unable to connect to vLLM server at {self.llm_url} after {timeout} seconds.\n"
            f"Please ensure the vLLM server is running at {self.llm_url}.\n"
            f"You can start it manually using: ./scripts/run_vllm_server.sh"
        )

    def stream_response(self, prompt, grammar=None, temperature=0.0, display_output=True):
        with Agent._llm_lock:
            user_message = prompt.split("<|im_start|>user\n")[-1].split("<|im_end|>")[0].strip()
            request_messages = []
            if self.agent_prompt:
                request_messages.append({"role": "system", "content": self.agent_prompt})
            request_messages.append({"role": "user", "content": user_message})
            self._logger.debug(
                f"Sending chat request to vLLM/OpenAI client. Model={self.model_name}, temperature={temperature}\nUser message:\n{user_message[:500]}"
            )
            request_kwargs = {
                "model": self.model_name,
                "messages": request_messages,
                "temperature": temperature,
                "max_tokens": self.ctx_length,
            }
            # If a JSON schema grammar is provided, use OpenAI-compatible response_format
            if grammar:
                try:
                    schema_dict = json.loads(grammar) if isinstance(grammar, str) else grammar
                    request_kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "structured_output",
                            "schema": schema_dict,
                            "strict": True,
                        },
                    }
                    # Also include vLLM-specific guided_json for broader compatibility
                    request_kwargs["extra_body"] = {"guided_json": schema_dict}
                except Exception as e:
                    self._logger.error(f"Failed to parse grammar for response_format: {e}")

            try:
                completion = self.client.chat.completions.create(**request_kwargs)
            except Exception as e:
                msg = str(e)
                # Fallback: drop structured output if server rejects the schema/guided_json
                if ("400" in msg or "Bad Request" in msg or "Grammar error" in msg or "response_format" in msg or "guided_json" in msg) and grammar:
                    self._logger.warning("Chat request failed with structured output; retrying without response_format/guided_json")
                    request_kwargs.pop("response_format", None)
                    if isinstance(request_kwargs.get("extra_body"), dict):
                        request_kwargs["extra_body"].pop("guided_json", None)
                        if not request_kwargs["extra_body"]:
                            request_kwargs.pop("extra_body", None)
                    try:
                        completion = self.client.chat.completions.create(**request_kwargs)
                    except Exception as e2:
                        self._logger.error(f"vLLM chat request failed after fallback: {e2}", exc_info=True)
                        return ""
                else:
                    self._logger.error(f"vLLM chat request failed: {e}", exc_info=True)
                    return ""

            response_text = completion.choices[0].message.content if completion.choices else ""
            if display_output and self.response_handler:
                self.response_handler.add_response(response_text)
                self.response_handler.end_response()
            return response_text

    def stream_image_response(
        self,
        prompt: str,
        image_b64: str,
        *,
        grammar: str | None = None,
        temperature: float = 0.0,
        display_output: bool = True,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        """
        Send a multimodal (text + image) request. Prefer the OpenAI Responses API
        for images to avoid HF chat template issues that some VL models have when
        `content` is a list.
        """
        # 1 – extract the user text
        try:
            user_message = prompt.split("<|im_start|>user\n")[-1].split("<|im_end|>")[0].strip()
        except Exception:
            user_message = prompt

        # 2 – prepare base64
        raw_b64 = self._extract_raw_base64(image_b64)

        # 3 – optionally reinforce the existence of the image
        modified_message = user_message
        if any(tok in user_message.lower() for tok in ("tool", "instrument")) and "image" not in user_message.lower():
            modified_message += " (Please look at the surgery image attached to this message.)"

        # 4 – build Inputs for Responses API
        if self._is_qwen_vl():
            image_part = {
                "type": "input_image",
                "image_data": {"data": raw_b64, "mime_type": "image/jpeg"},
            }
        else:
            image_part = {
                "type": "input_image",
                "image_url": {"url": f"data:image/jpeg;base64,{raw_b64}"},
            }

        responses_input = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": modified_message},
                    image_part,
                ],
            }
        ]

        answer = ""
        with Agent._llm_lock:
            req = {
                "model": self.model_name,
                "input": responses_input,
                "temperature": temperature,
                "max_output_tokens": self.ctx_length,
            }
            # Prepare extra knobs for vLLM
            extra_body: dict[str, Any] = {}
            if self._is_qwen_vl():
                # Help Qwen‑VL with larger vision inputs
                extra_body["mm_processor_kwargs"] = {"max_pixels": 12845056}

            if grammar:
                try:
                    schema_dict = json.loads(grammar) if isinstance(grammar, str) else grammar
                    # For Responses API, avoid response_format (client may not support it)
                    # Use vLLM-specific guided_json only.
                    extra_body["guided_json"] = schema_dict
                except Exception as e:
                    self._logger.error(f"Failed to parse grammar for Responses API (image): {e}")

            if extra_body:
                req["extra_body"] = extra_body
            elif extra_body is not None:
                req["extra_body"] = extra_body

            self._logger.debug("Multimodal request via Responses API (%s)…", self.model_name)
            try:
                result = self.client.responses.create(**req)
                answer = getattr(result, "output_text", None) or ""
                if not answer:
                    # Generic fallback parsing
                    data = result.model_dump() if hasattr(result, "model_dump") else None
                    if isinstance(data, dict):
                        if "output_text" in data:
                            answer = data["output_text"]
                        elif isinstance(data.get("output"), list) and data["output"]:
                            first = data["output"][0]
                            if isinstance(first, dict) and isinstance(first.get("content"), list) and first["content"]:
                                c0 = first["content"][0]
                                if isinstance(c0, dict):
                                    answer = c0.get("text", "")
            except Exception as e:
                # Fallback to chat.completions (older path)
                self._logger.warning(
                    f"Responses API failed for multimodal: {e}. Falling back to chat.completions."
                )
                messages: list[dict[str, Any]] = []
                if self.agent_prompt:
                    messages.append({"role": "system", "content": self.agent_prompt})
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": modified_message},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{raw_b64}"},
                            },
                        ],
                    }
                )

                request_kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": self.ctx_length,
                }
                if grammar:
                    try:
                        schema_dict = json.loads(grammar) if isinstance(grammar, str) else grammar
                        request_kwargs["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "structured_output",
                                "schema": schema_dict,
                                "strict": True,
                            },
                        }
                        request_kwargs["extra_body"] = {"guided_json": schema_dict}
                    except Exception as e2:
                        self._logger.error(f"Failed to parse grammar (fallback path): {e2}")
                elif extra_body is not None:
                    request_kwargs["extra_body"] = extra_body

                try:
                    res2 = self.client.chat.completions.create(**request_kwargs)
                    answer = res2.choices[0].message.content if res2.choices else ""
                except Exception as e3:
                    msg2 = str(e3)
                    if ("400" in msg2 or "Bad Request" in msg2 or "Grammar error" in msg2 or "response_format" in msg2 or "guided_json" in msg2) and grammar:
                        self._logger.warning("Fallback chat failed with structured output; retrying without response_format/guided_json")
                        request_kwargs.pop("response_format", None)
                        if isinstance(request_kwargs.get("extra_body"), dict):
                            request_kwargs["extra_body"].pop("guided_json", None)
                            if not request_kwargs["extra_body"]:
                                request_kwargs.pop("extra_body", None)
                        try:
                            res2 = self.client.chat.completions.create(**request_kwargs)
                            answer = res2.choices[0].message.content if res2.choices else ""
                        except requests.exceptions.Timeout:
                            self._logger.error("vLLM request timed out (fallback without schema)")
                            raise TimeoutError("Model request timed out") from None
                        except Exception:
                            self._logger.exception("vLLM multimodal request failed (fallback without schema)")
                            return ""
                    else:
                        if isinstance(e3, requests.exceptions.Timeout):
                            self._logger.error("vLLM request timed out (fallback)")
                            raise TimeoutError("Model request timed out") from None
                        self._logger.exception("vLLM multimodal request failed (fallback)")
                        return ""

        if display_output and self.response_handler:
            self.response_handler.add_response(answer)
            self.response_handler.end_response()
        return answer

    @staticmethod
    def _extract_raw_base64(data_uri: str) -> str:
        """
        Accept either a complete “data:image/…;base64,AAAA” URI *or* the bare
        base‑64 blob and always return the raw part (AAAA…).
        """
        if data_uri.startswith("data:image/"):
            return data_uri.split(",", 1)[1]
        return data_uri

    def generate_prompt(self, text, chat_history):
        system_prompt = f"{self.bot_rule_prefix}\n{self.agent_prompt}\n{self.end_token}"
        user_prompt = f"\n{self.user_prefix}\n{text}\n{self.end_token}"
        token_usage = self.calculate_token_usage(system_prompt + user_prompt)
        chat_prompt = self.create_conversation_str(chat_history, token_usage)
        prompt = system_prompt + chat_prompt + user_prompt
        prompt += f"\n{self.bot_prefix}\n"
        return prompt

    def create_conversation_str(self, chat_history, token_usage, conversation_length=2):
        total_tokens = token_usage
        msg_hist = []
        for user_msg, bot_msg in chat_history[:-1][-conversation_length:][::-1]:
            if bot_msg:
                bot_msg_str = f"\n{self.bot_prefix}\n{bot_msg}\n{self.end_token}"
                bot_tokens = self.calculate_token_usage(bot_msg_str)
                if total_tokens + bot_tokens > self.max_prompt_tokens:
                    break
                total_tokens += bot_tokens
                msg_hist.append(bot_msg_str)
            if user_msg:
                user_msg_str = f"\n{self.user_prefix}\n{user_msg}\n{self.end_token}"
                user_tokens = self.calculate_token_usage(user_msg_str)
                if total_tokens + user_tokens > self.max_prompt_tokens:
                    break
                total_tokens += user_tokens
                msg_hist.append(user_msg_str)
        return "".join(msg_hist[::-1])

    def calculate_token_usage(self, text):
        return len(self.tokenizer.encode(text))

    @abstractmethod
    def process_request(self, input_data, chat_history):
        pass

    def append_json_to_file(self, json_object, file_path):
        try:
            if not os.path.isfile(file_path):
                with open(file_path, 'w') as f:
                    json.dump([json_object], f, indent=2)
            else:
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = []
                if not isinstance(data, list):
                    data = []
                data.append(json_object)
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
        except Exception as e:
            self._logger.error(f"append_json_to_file error: {e}", exc_info=True)

    # ========================================================================
    # MESSAGE BUS INTEGRATION
    # ========================================================================

    def _start_message_listener(self):
        """Start background thread to listen for messages from message bus"""
        def message_listener():
            self._logger.debug(f"{self.agent_name} message listener starting...")
            while True:
                try:
                    message = self.message_queue.get(timeout=1)
                    self._handle_incoming_message(message)
                except queue.Empty:
                    continue
                except Exception as e:
                    self._logger.error(f"Error in message listener: {e}", exc_info=True)

        self._message_listener_thread = threading.Thread(
            target=message_listener,
            daemon=True,
            name=f"{self.agent_name}_MessageListener"
        )
        self._message_listener_thread.start()
        self._logger.info(f"{self.agent_name} message listener started")

    def _handle_incoming_message(self, message):
        """
        Handle incoming messages from message bus.
        Default behavior: call handle_{action} method if it exists.
        Override in subclasses for custom message handling.

        Args:
            message: AgentMessage object from message bus
        """
        handler_name = f"handle_{message.action}"

        if hasattr(self, handler_name):
            handler = getattr(self, handler_name)
            try:
                self._logger.debug(f"Handling message: {message.action} from {message.sender_agent}")
                handler(message)
            except Exception as e:
                self._logger.error(
                    f"Error handling {message.action} from {message.sender_agent}: {e}",
                    exc_info=True
                )
        else:
            self._logger.debug(
                f"No handler for action '{message.action}' (would need method '{handler_name}')"
            )
