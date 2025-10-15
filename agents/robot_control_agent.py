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

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

import zmq
from pydantic import BaseModel, ValidationError

from .base_agent import Agent


class _RobotCommandResult(BaseModel):
    command: Literal["start", "pause", "reset", "unknown"]
    natural_response: Optional[str] = None


class RobotControlAgent(Agent):
    """
    Interprets high-level user requests and relays start/pause/reset commands
    to the policy process over ZeroMQ.
    """

    _DEFAULT_SUCCESS_RESPONSES = {
        "start": "Starting the robot policy now.",
        "pause": "Pausing the robot policy.",
        "reset": "Resetting the robot policy to the home pose.",
    }

    def __init__(self, settings_path: str, response_handler=None):
        super().__init__(settings_path, response_handler)
        self._logger = logging.getLogger(__name__)

        self._control_endpoint: str = self.agent_settings.get(
            "control_endpoint", "tcp://localhost:5556"
        )
        self._command_timeout_s: float = float(
            self.agent_settings.get("command_timeout_s", 2.0)
        )
        self._retry_attempts: int = max(
            0, int(self.agent_settings.get("retry_attempts", 1))
        )
        self._linger_ms: int = int(self.agent_settings.get("linger_ms", 0))
        self._include_ack_in_response: bool = bool(
            self.agent_settings.get("include_ack_in_response", False)
        )
        self._unknown_response: str = self.agent_settings.get(
            "unknown_response",
            "I did not understand whether to start, pause, or reset the robot.",
        )
        self._error_template: str = self.agent_settings.get(
            "error_response_template",
            "I tried to {command} the robot, but {error}.",
        )

        success_overrides: Dict[str, str] = (
            self.agent_settings.get("success_responses", {}) or {}
        )
        self._success_responses: Dict[str, str] = {
            **RobotControlAgent._DEFAULT_SUCCESS_RESPONSES,
            **{k: v for k, v in success_overrides.items() if isinstance(v, str)},
        }

        self._context = zmq.Context.instance()

    def process_request(
        self,
        text: str,
        chat_history: List[List[Optional[str]]],
        visual_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            parsed = self._classify_command(text, chat_history)
        except Exception as exc:
            self._logger.error("Robot command classification failed: %s", exc, exc_info=True)
            parsed = None

        if not parsed:
            response_text = self._unknown_response
            return {
                "name": "RobotControlAgent",
                "response": response_text,
                "command": "unknown",
            }

        if parsed.command == "unknown":
            response_text = parsed.natural_response or self._unknown_response
            return {
                "name": "RobotControlAgent",
                "response": response_text,
                "command": "unknown",
            }

        success, controller_reply = self._dispatch_command(parsed.command)
        response_text = self._build_response(parsed.command, success, controller_reply)

        payload: Dict[str, Any] = {
            "name": "RobotControlAgent",
            "response": response_text,
            "command": parsed.command,
        }
        if controller_reply is not None:
            payload["controller_reply"] = controller_reply
        if not success:
            payload["error"] = controller_reply
        return payload

    def _classify_command(
        self, text: str, chat_history: List[List[Optional[str]]]
    ) -> Optional[_RobotCommandResult]:
        if not self.grammar:
            self._logger.error("RobotControlAgent requires a grammar for structured output.")
            return None

        prompt = self.generate_prompt(text, chat_history or [])
        raw_json = self.stream_response(
            prompt=prompt,
            grammar=self.grammar,
            temperature=0.0,
            display_output=False,
        )

        if not raw_json:
            self._logger.warning("Robot control LLM returned empty output.")
            return None

        raw_json = raw_json.strip()
        try:
            return _RobotCommandResult.model_validate_json(raw_json)
        except ValidationError as ve:
            self._logger.error(
                "Failed to parse robot control output: %s | raw=%s", ve, raw_json
            )
        except ValueError as ve:
            self._logger.error("Invalid JSON from robot control LLM: %s", ve)
        return None

    def _dispatch_command(self, command: str) -> Tuple[bool, Optional[str]]:
        attempts = self._retry_attempts + 1
        last_error: Optional[str] = None

        for attempt in range(1, attempts + 1):
            socket = self._context.socket(zmq.REQ)
            try:
                socket.setsockopt(zmq.LINGER, self._linger_ms)
                timeout_ms = max(int(self._command_timeout_s * 1000), 1)
                socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
                socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
                socket.connect(self._control_endpoint)

                self._logger.info(
                    "Sending robot command '%s' to %s (attempt %d/%d)",
                    command,
                    self._control_endpoint,
                    attempt,
                    attempts,
                )
                socket.send_string(command)
                reply = socket.recv_string()
                self._logger.info(
                    "Robot controller replied to '%s': %s", command, reply
                )
                return True, reply
            except zmq.Again:
                last_error = (
                    f"timeout waiting for acknowledgement on {self._control_endpoint}"
                )
                self._logger.warning(
                    "Timeout sending robot command '%s' (attempt %d/%d)",
                    command,
                    attempt,
                    attempts,
                )
            except Exception as exc:
                last_error = str(exc)
                self._logger.error(
                    "Failed to send robot command '%s': %s (attempt %d/%d)",
                    command,
                    exc,
                    attempt,
                    attempts,
                    exc_info=True,
                )
            finally:
                try:
                    socket.close(linger=self._linger_ms)
                except Exception:
                    pass

        return False, last_error

    def _build_response(
        self, command: str, success: bool, controller_reply: Optional[str]
    ) -> str:
        if success:
            base_response = self._success_responses.get(
                command, f"Command '{command}' sent to the robot."
            )
            if self._include_ack_in_response and controller_reply:
                return f"{base_response} Controller replied: {controller_reply}."
            return base_response

        error_text = (
            controller_reply
            if controller_reply
            else "the controller did not respond in time"
        )
        try:
            return self._error_template.format(command=command, error=error_text)
        except Exception:
            return f"I tried to {command} the robot, but {error_text}."
