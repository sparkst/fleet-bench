"""OpenRouter provider: OpenAI-compatible chat over the OpenRouter API.

Key from the OPENROUTER_API_KEY environment variable (never committed).
"""

from __future__ import annotations

import os

from .openai_compat import OpenAICompatProvider

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Chat models used for Phase 2 (see configs/phase2.yaml). A model absent from a
# plan is reported as n/a by the runner, never skipped.
OPENROUTER_MODELS = [
    "deepseek/deepseek-chat-v3.1",
    "deepseek/deepseek-r1-0528",
    "qwen/qwen3-coder-30b-a3b-instruct",
    "z-ai/glm-4.6",
    "moonshotai/kimi-k2-0905",
]


class OpenRouterProvider(OpenAICompatProvider):
    name = "openrouter"

    def __init__(self, *, models: list[str] | None = None, rpm: float = 60.0, **kwargs):
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        super().__init__(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            models=models or list(OPENROUTER_MODELS),
            rpm=rpm,
            **kwargs,
        )

    def available_models(self) -> list[str]:
        if not self.api_key:
            # Reported (not skipped): the runner records n/a: key-missing.
            return []
        return list(self.models)
