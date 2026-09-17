"""Groq provider: OpenAI-compatible chat over the free Groq API.

Key from the GROQ_API_KEY environment variable (never committed). Groq enforces
a per-model requests-per-minute limit; the default here is conservative.
"""

from __future__ import annotations

import os

from .openai_compat import OpenAICompatProvider

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Chat models available on the Groq free plan (verified live 2026-09-17). A
# model absent from a plan is reported as n/a by the runner, never skipped.
GROQ_MODELS = [
    "allam-2-7b",
    "groq/compound",
    "groq/compound-mini",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]


class GroqProvider(OpenAICompatProvider):
    name = "groq"

    def __init__(self, *, models: list[str] | None = None, rpm: float = 20.0, **kwargs):
        api_key = os.environ.get("GROQ_API_KEY", "")
        super().__init__(
            base_url=GROQ_BASE_URL,
            api_key=api_key,
            models=models or list(GROQ_MODELS),
            rpm=rpm,
            **kwargs,
        )

    def available_models(self) -> list[str]:
        if not self.api_key:
            # Reported (not skipped): the runner records n/a: key-missing.
            return []
        return list(self.models)
