#!/usr/bin/env python3
"""HTTP client for OpenAI-compatible chat endpoints.

Works with llama.cpp server, LM Studio, vLLM, Ollama, SGLang and
anything else that speaks /v1/chat/completions.

Design decisions worth knowing:

* Prefill and decode speed come from the server's own ``timings`` block when it
  reports one (llama.cpp / LM Studio do). Otherwise they are derived from
  ``usage`` and wall-clock time, and the metric is flagged as ``client``.
  We never present client wall time as prefill -- it includes generation.
* Default HTTP timeout is 3600 s. A 128K prompt on a slow machine legitimately
  takes minutes; a 600 s timeout silently turns slow machines into failures.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Optional


class EngineError(RuntimeError):
    pass


class Client:
    """Minimal dependency-free chat client (stdlib only)."""

    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None,
                 timeout: float = 3600.0):
        base = base_url.rstrip("/")
        self.url = base + ("/chat/completions" if base.endswith("/v1")
                           else "/v1/chat/completions")
        self.base = base
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    # ------------------------------------------------------------------ basic
    def _post(self, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:500]
            raise EngineError(f"HTTP {e.code}: {body}") from None
        except urllib.error.URLError as e:
            raise EngineError(f"cannot reach {self.url}: {e.reason}") from None

    def models(self) -> list[str]:
        req = urllib.request.Request(self.base.rstrip("/") + "/models")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            raise EngineError(f"model list failed: {e}") from None
        return [m.get("id") for m in d.get("data", [])]

    # -------------------------------------------------------------- completion
    def chat(self, messages: list[dict], *, max_tokens: int = 512,
             temperature: float = 0.0, tools: Optional[list] = None,
             no_thinking: bool = True, extra: Optional[dict] = None) -> dict:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = tools
        if no_thinking:
            # Most Qwen-family templates honour this; servers that ignore it are unaffected.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if extra:
            payload.update(extra)

        t0 = time.time()
        d = self._post(payload)
        wall = time.time() - t0

        choice = (d.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = d.get("usage") or {}
        timings = d.get("timings") or {}

        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or
                     msg.get("reasoning") or "").strip()

        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        # server-reported throughput wins when present
        prefill_tps = timings.get("prompt_per_second")
        decode_tps = timings.get("predicted_per_second")
        speed_source = "server" if (prefill_tps or decode_tps) else "client"
        if not decode_tps and completion_tokens and wall > 0:
            decode_tps = completion_tokens / wall
        if not prefill_tps and prompt_tokens and timings.get("prompt_ms"):
            prefill_tps = prompt_tokens / (timings["prompt_ms"] / 1000.0)

        return {
            "content": content,
            "reasoning": reasoning,
            "text": content or reasoning,          # what to score
            "tool_calls": msg.get("tool_calls") or [],
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "prefill_tok_s": round(prefill_tps, 1) if prefill_tps else None,
            "decode_tok_s": round(decode_tps, 1) if decode_tps else None,
            "speed_source": speed_source,
            "wall_s": round(wall, 2),
        }

    # ------------------------------------------------------------ sanity check
    def health(self) -> tuple[bool, str]:
        try:
            ms = self.models()
        except EngineError as e:
            return False, str(e)
        if self.model not in ms:
            return False, f"model '{self.model}' not served; available: {ms}"
        return True, f"ok, model '{self.model}' present ({len(ms)} served)"


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)
