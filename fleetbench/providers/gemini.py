"""Gemini provider: OpenAI-compatible chat over Google's Gemini endpoint.

Key from the GEMINI_API_KEY environment variable (never committed). The Gemini
free tier's requests-per-day budget is tight, so the default RPM here is
conservative.
"""

from __future__ import annotations

import os

from .openai_compat import OpenAICompatProvider

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

# Chat models used for Phase 2 (see configs/phase2.yaml). A model absent from a
# plan is reported as n/a by the runner, never skipped.
GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]


class GeminiProvider(OpenAICompatProvider):
    name = "gemini"

    def __init__(self, *, models: list[str] | None = None, rpm: float = 15.0, **kwargs):
        api_key = os.environ.get("GEMINI_API_KEY", "")
        super().__init__(
            base_url=GEMINI_BASE_URL,
            api_key=api_key,
            models=models or list(GEMINI_MODELS),
            rpm=rpm,
            **kwargs,
        )

    def available_models(self) -> list[str]:
        if not self.api_key:
            # Reported (not skipped): the runner records n/a: key-missing.
            return []
        return list(self.models)
