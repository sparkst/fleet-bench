"""Codex provider via the logged-in ``codex`` CLI (``codex exec --json``).

Runs over ssh on a configured host label when one is set (the fleet shards the
Codex leg to a host where the CLI is logged in). Prompt on stdin. Uses
``--json`` for a machine-parseable transcript: the final agent message is the
answer and the usage event carries token counts. ``--skip-git-repo-check`` lets
it run outside a trusted git directory. The Codex plan serves gpt-5.6-sol only.
"""

from __future__ import annotations

import json

from .cli_base import CLIProvider

CODEX_MODELS = ["gpt-5.6-sol"]


class CodexCLIProvider(CLIProvider):
    name = "codex"
    base_command = ["codex", "exec", "--skip-git-repo-check", "--json", "--model", "{model}"]

    def __init__(self, *, models: list[str] | None = None, **kwargs):
        super().__init__(models=models or list(CODEX_MODELS), **kwargs)

    def parse_result(self, stdout: str, stderr: str) -> dict:
        texts: list[str] = []
        tokens_in = tokens_out = None
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            if etype == "item.completed":
                item = event.get("item") or {}
                if item.get("type") == "agent_message" and item.get("text"):
                    texts.append(item["text"])
            elif etype == "turn.completed":
                usage = event.get("usage") or {}
                tokens_in = usage.get("input_tokens")
                tokens_out = usage.get("output_tokens")
        return {
            "text": "\n".join(texts).strip(),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }
