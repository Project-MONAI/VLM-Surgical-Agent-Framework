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

"""
vLLM server warmup utility.

Performs startup tasks and caches the results per (llm_url, model_name)
so that multiple Agent instances sharing the same endpoint skip redundant work:

1. **Wait** for the vLLM ``/models`` endpoint to become reachable.
2. **Probe** whether the OpenAI *Responses API* (``/responses``) is available.
3. **Warm up** the model by sending a short text completion.

Set ``RESPONSE_API_USE=disable`` to force-disable the Responses API
(skips the probe and always uses ``chat.completions`` for multimodal).

Usage::

    from utils.vllm_warmup import vllm_warmup

    info = vllm_warmup(llm_url="http://localhost:8000/v1",
                       model_name="my-model")
    info.responses_api_supported   # bool
    info.server_ready              # bool
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional

import yaml
import requests
from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class WarmupResult:
    """Record returned (and cached) by :func:`vllm_warmup`."""
    llm_url: str
    model_name: str
    server_ready: bool = False
    responses_api_supported: bool = False
    warmup_response: str = ""


# ---------------------------------------------------------------------------
# Module-level cache  (keyed by (llm_url, model_name))
# ---------------------------------------------------------------------------

_cache: Dict[tuple, WarmupResult] = {}
_cache_lock = Lock()


def get_cached(llm_url: str, model_name: str) -> Optional[WarmupResult]:
    with _cache_lock:
        return _cache.get((llm_url, model_name))


# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------

def wait_for_server(llm_url: str, timeout: int = 120) -> bool:
    """Block until ``GET <llm_url>/models`` returns 200 or *timeout* expires."""
    check_url = f"{llm_url}/models"
    for attempt in range(timeout):
        try:
            r = requests.get(check_url, timeout=5)
            if r.status_code == 200:
                logger.info("Connected to vLLM server at %s", llm_url)
                return True
        except Exception as exc:
            if attempt % 10 == 0:
                logger.info(
                    "Waiting for vLLM server (attempt %d/%d): %s",
                    attempt + 1, timeout, exc,
                )
            else:
                logger.debug(
                    "Waiting for vLLM server (attempt %d/%d): %s",
                    attempt + 1, timeout, exc,
                )
        time.sleep(1)

    logger.error(
        "Unable to connect to vLLM server at %s after %d seconds", llm_url, timeout,
    )
    return False


def probe_responses_api(llm_url: str) -> bool:
    """Return *True* when the server exposes and accepts requests to the Responses API."""
    status: int | None = None
    try:
        r = requests.post(f"{llm_url}/responses", json={}, timeout=5)
        status = r.status_code
        # Only treat as supported when the server returns 2xx (e.g. 200/201).
        # 400 Bad Request means the endpoint exists but our format isn't accepted – use chat.completions.
        supported = 200 <= status < 300
    except Exception:
        supported = False
    logger.info(
        "Responses API probe: %s (url=%s/responses%s)",
        "supported" if supported else "not supported – will use chat.completions for multimodal",
        llm_url,
        f", status={status}" if status is not None else "",
    )
    return supported


def warmup_completion(llm_url: str, model_name: str) -> str:
    """Send a trivial completion so the model weights are loaded / warm."""
    client = OpenAI(api_key="EMPTY", base_url=llm_url)
    try:
        t0 = time.time()
        res = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            temperature=0.0,
        )
        elapsed = time.time() - t0
        text = res.choices[0].message.content if res.choices else ""
        logger.info("Warmup completion finished in %.1fs", elapsed)
        return text
    except Exception as exc:
        logger.warning("Warmup completion failed (non-fatal): %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Config resolution (mirrors Agent.load_settings precedence)
# ---------------------------------------------------------------------------

def _load_global_config() -> dict:
    """Load configs/global.yaml using the same lookup the agents use."""
    env_path = os.environ.get("VLLM_GLOBAL_CONFIG")
    candidates = []
    if env_path:
        candidates.append(env_path)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.path.join(repo_root, "configs", "global.yaml"))
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
    return {}


def _resolve_config(
    llm_url: str | None,
    model_name: str | None,
) -> tuple[str, str]:
    """
    Resolve *llm_url* and *model_name* using the same precedence as
    ``Agent.load_settings``: explicit arg > env var > global.yaml > default.
    """
    global_cfg = _load_global_config()

    if not llm_url:
        llm_url = (
            os.environ.get("VLLM_URL")
            or global_cfg.get("llm_url")
            or "http://localhost:8000/v1"
        )

    if not model_name:
        model_name = (
            os.environ.get("VLLM_SERVED_MODEL_NAME")
            or os.environ.get("VLLM_MODEL_NAME")
            or global_cfg.get("served_model_name")
            or global_cfg.get("model_name")
            or "llama3.2"
        )

    return llm_url, model_name


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def vllm_warmup(
    llm_url: str | None = None,
    model_name: str | None = None,
    *,
    timeout: int = 120,
    skip_warmup_completion: bool = False,
) -> WarmupResult:
    """
    Run all warmup steps and cache the result.

    Parameters
    ----------
    llm_url : str, optional
        Base URL of the vLLM OpenAI-compatible API.
        Falls back to ``$VLLM_URL``, then ``configs/global.yaml``,
        then ``http://localhost:8000/v1``.
    model_name : str, optional
        Model identifier used for the warmup completion.
        Falls back to ``$VLLM_SERVED_MODEL_NAME``, ``$VLLM_MODEL_NAME``,
        then ``configs/global.yaml``, then ``"llama3.2"``.
    timeout : int
        Seconds to wait for the server to become reachable.
    skip_warmup_completion : bool
        If *True*, skip the warmup completion request.
    """
    llm_url, model_name = _resolve_config(llm_url, model_name)

    cached = get_cached(llm_url, model_name)
    if cached is not None:
        logger.debug("Returning cached warmup result for %s / %s", llm_url, model_name)
        return cached

    result = WarmupResult(llm_url=llm_url, model_name=model_name)

    result.server_ready = wait_for_server(llm_url, timeout=timeout)
    if not result.server_ready:
        with _cache_lock:
            _cache[(llm_url, model_name)] = result
        return result

    # RESPONSE_API_USE=disable  →  skip probe, force chat.completions
    if os.environ.get("RESPONSE_API_USE", "").lower() == "disable":
        logger.info("Responses API disabled via RESPONSE_API_USE=disable")
        result.responses_api_supported = False
    else:
        result.responses_api_supported = probe_responses_api(llm_url)

    if not skip_warmup_completion:
        result.warmup_response = warmup_completion(llm_url, model_name)

    with _cache_lock:
        _cache[(llm_url, model_name)] = result

    logger.info(
        "vLLM warmup complete — server_ready=%s, responses_api=%s",
        result.server_ready,
        result.responses_api_supported,
    )
    return result
