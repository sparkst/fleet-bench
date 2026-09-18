"""OpenRouter, Gemini, and Cloudflare providers: thin OpenAICompatProvider
subclasses added for Phase 2 (see configs/phase2.yaml).

Network is stubbed by replacing the shared module opener; no real request is
made. Mirrors the mocking style of tests/test_openai_compat.py.
"""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest
import yaml

from fleetbench.providers import PROVIDERS, openai_compat as oc
from fleetbench.providers.base import ProviderError
from fleetbench.providers.cloudflare import CloudflareProvider
from fleetbench.providers.gemini import GeminiProvider
from fleetbench.providers.openrouter import OpenRouterProvider
from fleetbench.runner import RunConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _install_success_opener(monkeypatch, captured, text="hi there"):
    payload = {"choices": [{"message": {"content": text}}], "usage": {}}

    class _Opener:
        def open(self, req, timeout):
            captured["url"] = req.full_url
            captured["auth"] = req.get_header("Authorization")
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResp(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(oc, "_OPENER", _Opener())


def _install_429_opener(monkeypatch):
    class _Opener:
        def open(self, req, timeout):
            raise urllib.error.HTTPError(
                req.full_url, 429, "too many requests", {}, io.BytesIO(b"slow down")
            )

    monkeypatch.setattr(oc, "_OPENER", _Opener())


# --- OpenRouter -------------------------------------------------------------


def test_openrouter_base_url_and_auth(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    captured = {}
    _install_success_opener(monkeypatch, captured)
    p = OpenRouterProvider()
    assert p.base_url == "https://openrouter.ai/api/v1"
    out = p.call("deepseek/deepseek-chat-v3.1", "hi")
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["auth"] == "Bearer or-key"
    assert captured["body"]["model"] == "deepseek/deepseek-chat-v3.1"
    assert out.text == "hi there"


def test_openrouter_429_triggers_na_after_retries(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    _install_429_opener(monkeypatch)
    p = OpenRouterProvider(sleep=lambda s: None)
    completion = p.run_item("deepseek/deepseek-chat-v3.1", "hi")
    assert completion.is_na
    assert "error after retries" in completion.na_reason


def test_openrouter_available_models_empty_without_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    p = OpenRouterProvider()
    assert p.available_models() == []


# --- Gemini -------------------------------------------------------------


def test_gemini_base_url_and_auth(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    captured = {}
    _install_success_opener(monkeypatch, captured)
    p = GeminiProvider()
    assert p.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    out = p.call("gemini-2.5-flash", "hi")
    assert captured["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    assert captured["auth"] == "Bearer g-key"
    assert captured["body"]["model"] == "gemini-2.5-flash"
    assert out.text == "hi there"


def test_gemini_429_triggers_na_after_retries(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    _install_429_opener(monkeypatch)
    p = GeminiProvider(sleep=lambda s: None)
    completion = p.run_item("gemini-2.5-flash", "hi")
    assert completion.is_na
    assert "error after retries" in completion.na_reason


def test_gemini_available_models_empty_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    p = GeminiProvider()
    assert p.available_models() == []


# --- Cloudflare -----------------------------------------------------------


def test_cloudflare_base_url_and_auth(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-key")
    captured = {}
    _install_success_opener(monkeypatch, captured)
    p = CloudflareProvider()
    assert p.base_url == "https://api.cloudflare.com/client/v4/accounts/acct123/ai/v1"
    out = p.call("@cf/openai/gpt-oss-120b", "hi")
    assert captured["url"] == (
        "https://api.cloudflare.com/client/v4/accounts/acct123/ai/v1/chat/completions"
    )
    assert captured["auth"] == "Bearer cf-key"
    assert captured["body"]["model"] == "@cf/openai/gpt-oss-120b"
    assert out.text == "hi there"


def test_cloudflare_429_triggers_na_after_retries(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-key")
    _install_429_opener(monkeypatch)
    p = CloudflareProvider(sleep=lambda s: None)
    completion = p.run_item("@cf/openai/gpt-oss-120b", "hi")
    assert completion.is_na
    assert "error after retries" in completion.na_reason


def test_cloudflare_available_models_empty_without_key(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    p = CloudflareProvider()
    assert p.available_models() == []


def test_cloudflare_missing_account_id_raises_loudly(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-key")
    with pytest.raises(RuntimeError, match="CLOUDFLARE_ACCOUNT_ID"):
        CloudflareProvider()


# --- configs/phase2.yaml ---------------------------------------------------


def test_phase2_yaml_parses_via_run_config():
    raw = yaml.safe_load((REPO_ROOT / "configs" / "phase2.yaml").read_text(encoding="utf-8"))
    cfg = RunConfig.from_dict(raw)
    assert cfg.suite == "phase2"
    assert cfg.datasets["ifeval"] == 100
    assert cfg.datasets["humaneval"] == 164
    assert cfg.datasets["trec6"] == 200
    assert cfg.datasets["gsm8k"] == 100
    assert cfg.datasets["sanity"] == 30
    assert cfg.datasets["sdlc"] == 18
    assert cfg.judge["provider"] == "cloudflare"
    assert cfg.judge["calibration_provider"] == "openrouter"


def test_phase2_yaml_providers_all_registered():
    raw = yaml.safe_load((REPO_ROOT / "configs" / "phase2.yaml").read_text(encoding="utf-8"))
    cfg = RunConfig.from_dict(raw)
    names = {p["name"] for p in cfg.providers}
    names.add(cfg.judge["provider"])
    names.add(cfg.judge["calibration_provider"])
    for name in names:
        assert name in PROVIDERS, f"provider {name!r} not registered in build_provider"
