"""Anthropic provider via the logged-in ``claude`` CLI (print mode).

Runs on the local host (no ssh). Prompt on stdin, answer on stdout.
"""

from __future__ import annotations

from .cli_base import CLIProvider

ANTHROPIC_MODELS = [
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-4-5-20251001",
]


class AnthropicCLIProvider(CLIProvider):
    name = "anthropic"
    base_command = ["claude", "-p", "--model", "{model}"]

    def __init__(self, *, models: list[str] | None = None, **kwargs):
        super().__init__(models=models or list(ANTHROPIC_MODELS), **kwargs)
