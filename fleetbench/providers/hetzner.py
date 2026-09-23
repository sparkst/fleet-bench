"""Hetzner Experiments provider: OpenAI-compatible chat, thinking off by default.

Key from HETZNER_EXPERIMENTS_TOKEN. The Qwen models on this endpoint return
empty content when their thinking mode is on, so it is disabled by default via
the chat-template kwargs the endpoint honors.

Every call appends one JSONL line to a ledger so the daily self-budget (500
requests/day) can be counted. The ledger path is read from an environment
variable and defaults to a local results file; no host-specific path is baked
into this public repository.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

from .base import Completion
from .openai_compat import OpenAICompatProvider

HETZNER_BASE_URL = "https://inference.hetzner.com/api/v1"

# Models verified live 2026-09-17 on the Experiments endpoint.
HETZNER_MODELS = [
    "Qwen/Qwen3.6-35B-A3B-FP8",
    "Qwen3.8-27B",
]

# Turning off Qwen thinking. The endpoint reads OpenAI-style extra body; this is
# the documented toggle. Kept in one place so it is easy to adjust if the
# endpoint's accepted key changes.
THINKING_OFF_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}

_LEDGER_LOCK = threading.Lock()


class HetznerProvider(OpenAICompatProvider):
    name = "hetzner"

    def __init__(
        self,
        *,
        models: list[str] | None = None,
        rpm: float = 0.0,
        ledger_path: Optional[str] = None,
        daily_budget: int = 500,
        **kwargs,
    ):
        api_key = os.environ.get("HETZNER_EXPERIMENTS_TOKEN", "")
        super().__init__(
            base_url=HETZNER_BASE_URL,
            api_key=api_key,
            models=models or list(HETZNER_MODELS),
            rpm=rpm,
            extra_body=dict(THINKING_OFF_EXTRA_BODY),
            **kwargs,
        )
        self.daily_budget = daily_budget
        self.ledger_path = (
            ledger_path
            or os.environ.get("HETZNER_LEDGER_PATH")
            or str(Path("results") / "hetzner-ledger.jsonl")
        )

    def available_models(self) -> list[str]:
        if not self.api_key:
            return []
        return list(self.models)

    def call(self, model: str, prompt: str) -> Completion:
        completion = super().call(model, prompt)
        self._write_ledger(model, completion)
        return completion

    def _write_ledger(self, model: str, completion: Completion) -> None:
        # One line per successful call. The fleet counts these against the
        # per-host daily budget. Fields are stable; extend, do not rename.
        line = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "provider": self.name,
            "model": model,
            "endpoint": self.base_url,
            "tokens_in": completion.tokens_in,
            "tokens_out": completion.tokens_out,
        }
        path = Path(self.ledger_path)
        with _LEDGER_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line) + "\n")
