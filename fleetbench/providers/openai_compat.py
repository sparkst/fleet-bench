"""OpenAI-compatible chat client, shared by the HTTP providers (Groq, Hetzner).

Uses stdlib ``urllib`` with a no-redirect opener so the bearer token is never
replayed to a redirect Location host (a known footgun: urllib auto-follows a
301/302/303 POST and re-sends the Authorization header). Provider endpoints do
not redirect in normal operation; refusing to follow one is the safe default.

No third-party HTTP dependency, so a clone runs with only PyYAML installed.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Optional

from ..scrub import scrub
from .base import Completion, Provider, ProviderError


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None  # never follow; do not replay the bearer to a new host


_OPENER = urllib.request.build_opener(_NoRedirect)


class OpenAICompatProvider(Provider):
    """A chat provider speaking the OpenAI ``/chat/completions`` shape.

    Subclasses (or callers) set ``base_url``, ``api_key``, ``models`` and, for
    models that need it, ``extra_body`` (for example Qwen's thinking toggle).
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        models: list[str],
        rpm: float = 0.0,
        concurrency: int = 1,
        timeout_s: float = 120.0,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        extra_body: Optional[dict] = None,
        user_agent: str = "fleet-bench/0.1 (+https://github.com/sparkst/fleet-bench)",
        **kwargs,
    ):
        super().__init__(rpm=rpm, concurrency=concurrency, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.models = list(models)
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_body = dict(extra_body or {})
        self.user_agent = user_agent

    def available_models(self) -> list[str]:
        # Configured model list. The runner cross-checks each requested model
        # against this so a model not on the plan is reported, not skipped.
        return list(self.models)

    def _payload(self, model: str, prompt: str) -> dict:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        body.update(self.extra_body)
        return body

    def call(self, model: str, prompt: str) -> Completion:
        url = f"{self.base_url}/chat/completions"
        data = json.dumps(self._payload(model, prompt)).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        # A default urllib User-Agent is blocked by some providers' edge (for
        # example Cloudflare 403 code 1010). Send an explicit client UA.
        req.add_header("User-Agent", self.user_agent)
        req.add_header("Accept", "application/json")
        started = time.monotonic()
        try:
            with _OPENER.open(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
            latency = time.monotonic() - started
        except urllib.error.HTTPError as exc:
            status = exc.code
            retryable = status == 429 or 500 <= status < 600
            # Scrub account/org identifiers from the upstream body: it may be
            # persisted to a public artifact as an n/a reason.
            detail = scrub(_safe_body(exc))
            raise ProviderError(
                f"HTTP {status} from {self.name}: {detail}",
                retryable=retryable,
                status=status,
            )
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProviderError(f"network error from {self.name}: {exc}", retryable=True)

        parsed = json.loads(raw)
        text, tin, tout = _extract(parsed)
        return Completion(
            text=text,
            latency_s=latency,
            tokens_in=tin,
            tokens_out=tout,
            raw_meta={"finish_reason": _finish_reason(parsed)},
        )


def _safe_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8")[:400]
    except Exception:
        return "<no body>"


def _extract(parsed: dict) -> tuple[str, Optional[int], Optional[int]]:
    choices = parsed.get("choices") or []
    text = ""
    if choices:
        message = choices[0].get("message") or {}
        text = message.get("content") or ""
    usage = parsed.get("usage") or {}
    return (
        text,
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
    )


def _finish_reason(parsed: dict) -> Optional[str]:
    choices = parsed.get("choices") or []
    if choices:
        return choices[0].get("finish_reason")
    return None
