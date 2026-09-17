"""Copilot provider via the logged-in ``copilot`` CLI (non-interactive).

Runs over ssh on a configured host label when one is set. The Copilot CLI takes
the prompt as the ``-p`` argument (not stdin), ``-s`` prints only the agent
response, and ``--allow-all-tools`` is required for non-interactive mode. The CLI
served gpt-5.4 and gpt-5-mini. Premium-request budget is enforced by the runner
through caps.yaml. In silent mode the CLI does not print token counts, so token
burn is reported as n/a for this provider.
"""

from __future__ import annotations

from .cli_base import CLIProvider

COPILOT_MODELS = ["gpt-5.4", "gpt-5-mini"]


class CopilotCLIProvider(CLIProvider):
    name = "copilot"
    base_command = [
        "copilot", "-s", "--allow-all-tools", "--model", "{model}", "-p", "{prompt}",
    ]

    def __init__(self, *, models: list[str] | None = None, **kwargs):
        super().__init__(models=models or list(COPILOT_MODELS), **kwargs)
