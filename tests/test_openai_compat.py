"""OpenAI-compatible client: payload, parse, and HTTP error mapping.

The network is stubbed by replacing the module opener; no real request is made.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from fleetbench.providers import openai_compat as oc
from fleetbench.providers.base import ProviderError
from fleetbench.providers.hetzner import HetznerProvider


def _provider(**kw):
    return oc.OpenAICompatProvider(
        base_url="https://example.test/v1", api_key="k", models=["m"], **kw
    )


def test_payload_includes_extra_body():
    p = _provider(extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    body = p._payload("m", "hi")
    assert body["model"] == "m"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"][0]["content"] == "hi"


def test_hetzner_sets_thinking_off():
    p = HetznerProvider()
    assert p.extra_body["chat_template_kwargs"]["enable_thinking"] is False


def test_extract_content_and_usage():
    parsed = {
        "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }
    text, tin, tout = oc._extract(parsed)
    assert text == "answer" and tin == 11 and tout == 7
    assert oc._finish_reason(parsed) == "stop"


class _FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_success_path(monkeypatch):
    payload = {"choices": [{"message": {"content": "hi there"}}], "usage": {}}
    captured = {}

    class _Opener:
        def open(self, req, timeout):
            captured["ua"] = req.get_header("User-agent")
            captured["auth"] = req.get_header("Authorization")
            return _FakeResp(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(oc, "_OPENER", _Opener())
    p = _provider()
    out = p.call("m", "prompt")
    assert out.text == "hi there"
    # An explicit, non-default User-Agent avoids edge blocks (Cloudflare 1010).
    assert captured["ua"] and "python-urllib" not in captured["ua"].lower()
    assert captured["auth"] == "Bearer k"


@pytest.mark.parametrize("status,retryable", [(429, True), (503, True), (400, False)])
def test_http_error_mapping(monkeypatch, status, retryable):
    class _Opener:
        def open(self, req, timeout):
            raise urllib.error.HTTPError(
                "https://example.test/v1", status, "err", {}, io.BytesIO(b"detail")
            )

    monkeypatch.setattr(oc, "_OPENER", _Opener())
    p = _provider()
    with pytest.raises(ProviderError) as exc:
        p.call("m", "prompt")
    assert exc.value.retryable is retryable
    assert exc.value.status == status
