"""Cloudflare Workers AI provider: OpenAI-compatible chat over Cloudflare.

Key from the CLOUDFLARE_API_TOKEN environment variable (never committed). The
account id is never hardcoded: it is read from CLOUDFLARE_ACCOUNT_ID at
construction time and interpolated into the base URL. A missing account id
raises loudly rather than silently proceeding with a placeholder.
"""

from __future__ import annotations

import os

from .base import Completion, ProviderError
from .openai_compat import OpenAICompatProvider

CLOUDFLARE_BASE_URL_TEMPLATE = (
    "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
)

# Chat models used for Phase 2 (see configs/phase2.yaml). A model absent from a
# plan is reported as n/a by the runner, never skipped.
CLOUDFLARE_MODELS = [
    "@cf/openai/gpt-oss-120b",
    "@cf/openai/gpt-oss-20b",
    "@cf/qwen/qwen2.5-coder-32b-instruct",
]


class CloudflareProvider(OpenAICompatProvider):
    name = "cloudflare"

    def __init__(self, *, models: list[str] | None = None, rpm: float = 0.0, **kwargs):
        account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        if not account_id:
            raise RuntimeError(
                "CLOUDFLARE_ACCOUNT_ID is not set: the cloudflare provider "
                "requires this environment variable to build its base URL. "
                "Set it before running (never hardcode it)."
            )
        api_key = os.environ.get("CLOUDFLARE_API_TOKEN", "")
        self._account_id = account_id
        super().__init__(
            base_url=CLOUDFLARE_BASE_URL_TEMPLATE.format(account_id=account_id),
            api_key=api_key,
            models=models or list(CLOUDFLARE_MODELS),
            rpm=rpm,
            **kwargs,
        )

    def available_models(self) -> list[str]:
        if not self.api_key:
            # Reported (not skipped): the runner records n/a: key-missing.
            return []
        return list(self.models)

    def call(self, model: str, prompt: str) -> Completion:
        # Belt-and-suspenders: the base scrub() catches generic id shapes, but
        # this provider knows its OWN account id (interpolated into the URL
        # and thus liable to be echoed back verbatim by a CF error body), so
        # strip that exact value from any error text before it can be
        # persisted to a public artifact.
        try:
            return super().call(model, prompt)
        except ProviderError as exc:
            if self._account_id and self._account_id in str(exc):
                scrubbed_msg = str(exc).replace(self._account_id, "[redacted-account-id]")
                raise ProviderError(scrubbed_msg, retryable=exc.retryable, status=exc.status) from exc
            raise
