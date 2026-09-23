"""Provider registry.

Maps a provider name to its class and builds an instance from run-config
overrides. CLI providers that shard to another host read their ssh host label
from the FLEETBENCH_CLI_SSH_HOST environment variable (never committed).
"""

from __future__ import annotations

import os
from typing import Optional

from .anthropic_cli import AnthropicCLIProvider
from .base import Completion, Provider, ProviderError, RateLimiter
from .codex_cli import CodexCLIProvider
from .copilot_cli import CopilotCLIProvider
from .groq import GroqProvider
from .hetzner import HetznerProvider

PROVIDERS: dict[str, type[Provider]] = {
    "groq": GroqProvider,
    "hetzner": HetznerProvider,
    "anthropic": AnthropicCLIProvider,
    "codex": CodexCLIProvider,
    "copilot": CopilotCLIProvider,
}

# Providers whose CLI is not on the local host; the fleet shards them over ssh.
_REMOTE_CLI = {"codex", "copilot"}


def build_provider(name: str, overrides: Optional[dict] = None) -> Provider:
    """Instantiate a provider by name with optional run-config overrides."""
    if name not in PROVIDERS:
        raise KeyError(f"unknown provider: {name}")
    kwargs = dict(overrides or {})
    if name in _REMOTE_CLI and "ssh_host" not in kwargs:
        ssh_host = os.environ.get("FLEETBENCH_CLI_SSH_HOST")
        if ssh_host:
            kwargs["ssh_host"] = ssh_host
        shell = os.environ.get("FLEETBENCH_CLI_SSH_SHELL")
        if shell and "remote_shell" not in kwargs:
            kwargs["remote_shell"] = shell
    return PROVIDERS[name](**kwargs)


def provider_names() -> list[str]:
    return list(PROVIDERS)


__all__ = [
    "Completion",
    "Provider",
    "ProviderError",
    "RateLimiter",
    "PROVIDERS",
    "build_provider",
    "provider_names",
]
