"""CLI provider: argv construction (local and ssh), output parse, error mapping.

Uses an injected exec_fn so no CLI binary is invoked.
"""

from __future__ import annotations

from fleetbench.providers.anthropic_cli import AnthropicCLIProvider
from fleetbench.providers.base import ProviderError
from fleetbench.providers.cli_base import CLIProvider
from fleetbench.providers.copilot_cli import CopilotCLIProvider


def _fake_exec(record):
    def _exec(argv, stdin, timeout_s):
        record["argv"] = argv
        record["stdin"] = stdin
        return record["ret"]
    return _exec


def test_local_argv_substitutes_model():
    rec = {"ret": (0, "hello", "")}
    p = AnthropicCLIProvider(exec_fn=_fake_exec(rec))
    out = p.call("claude-sonnet-5", "say hi")
    assert rec["argv"] == ["claude", "-p", "--model", "claude-sonnet-5"]
    assert rec["stdin"] == "say hi"
    assert out.text == "hello"
    assert out.raw_meta["cli_startup_included"] is True


def test_ssh_wrapping_wraps_in_login_shell():
    rec = {"ret": (0, "ok", "")}
    p = AnthropicCLIProvider(ssh_host="cli-host", exec_fn=_fake_exec(rec))
    p.call("claude-opus-5", "prompt text")
    assert rec["argv"][0] == "ssh"
    assert rec["argv"][1] == "cli-host"
    assert rec["argv"][2] == "--"
    # The remote command wraps the CLI in a login shell so it is on PATH.
    assert rec["argv"][3] == "zsh -lc 'claude -p --model claude-opus-5'"


def test_ssh_without_login_shell():
    rec = {"ret": (0, "ok", "")}
    p = AnthropicCLIProvider(ssh_host="cli-host", remote_shell=None, exec_fn=_fake_exec(rec))
    p.call("claude-opus-5", "hi")
    assert rec["argv"][3] == "claude -p --model claude-opus-5"


def test_prompt_as_arg_provider_puts_prompt_in_argv_not_stdin():
    rec = {"ret": (0, "answer", "")}
    p = CopilotCLIProvider(exec_fn=_fake_exec(rec))
    p.call("gpt-5-mini", "classify apple")
    assert "classify apple" in rec["argv"]
    assert "gpt-5-mini" in rec["argv"]
    assert rec["stdin"] == ""  # prompt goes as an argument, not stdin


def test_nonzero_exit_maps_to_error():
    rec = {"ret": (1, "", "bad model")}
    p = AnthropicCLIProvider(exec_fn=_fake_exec(rec))
    result = p.run_item("claude-sonnet-5", "hi")
    assert result.is_na
    assert "exit 1" in result.na_reason


def test_missing_binary_is_non_retryable_na():
    def _exec(argv, stdin, timeout_s):
        raise FileNotFoundError("codex")

    class P(CLIProvider):
        name = "x"
        base_command = ["codex", "exec", "--model", "{model}"]

    p = P(models=["m"], exec_fn=_exec)
    result = p.run_item("m", "hi")
    assert result.is_na
    assert "not found" in result.na_reason
