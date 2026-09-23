"""CLI provider base: drive a logged-in coding CLI as a benchmark provider.

Each CLI leg runs a subprocess (``claude -p``, ``codex exec``, ``copilot -p``),
feeding the prompt on stdin and reading the answer from stdout. Latency is
measured around the whole subprocess, so it INCLUDES CLI startup; reports
footnote this.

Host sharding (REQ-FB-05): when an ssh host label is configured the command is
run over ssh on that host instead of locally. The host is read from an
environment variable so no machine name is committed to this public repo.
"""

from __future__ import annotations

import shlex
import subprocess
import time
from typing import Callable, Optional

from .base import Completion, Provider, ProviderError

# Result of the subprocess seam: (returncode, stdout, stderr).
ExecResult = tuple[int, str, str]


def _default_exec(argv: list[str], stdin: str, timeout_s: float) -> ExecResult:
    proc = subprocess.run(
        argv,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return proc.returncode, proc.stdout, proc.stderr


class CLIProvider(Provider):
    """Base for CLI-backed providers.

    Subclasses set ``name``, ``models``, and ``base_command`` (an argv list with
    a literal ``{model}`` token where the model id goes). The prompt is passed on
    stdin. ``ssh_host`` (a label, resolved by the caller from env) reruns the
    command on a remote host over ssh.
    """

    base_command: list[str] = []

    def __init__(
        self,
        *,
        models: list[str],
        ssh_host: Optional[str] = None,
        remote_shell: Optional[str] = "zsh -lc",
        timeout_s: float = 300.0,
        exec_fn: Callable[[list[str], str, float], ExecResult] = _default_exec,
        rpm: float = 0.0,
        concurrency: int = 1,
        **kwargs,
    ):
        super().__init__(rpm=rpm, concurrency=concurrency, **kwargs)
        self.models = list(models)
        self.ssh_host = ssh_host
        # When running over ssh, wrap the command in a login shell so the CLI is
        # on PATH (a non-login ssh shell often lacks it). Set to None to disable.
        self.remote_shell = remote_shell
        self.timeout_s = timeout_s
        self._exec = exec_fn

    def available_models(self) -> list[str]:
        return list(self.models)

    def _uses_prompt_arg(self) -> bool:
        # A CLI that takes the prompt as an argument uses a {prompt} token in its
        # base_command; otherwise the prompt is fed on stdin.
        return any("{prompt}" in tok for tok in self.base_command)

    def _local_argv(self, model: str, prompt: str) -> list[str]:
        return [
            tok.replace("{model}", model).replace("{prompt}", prompt)
            for tok in self.base_command
        ]

    def _argv(self, model: str, prompt: str) -> list[str]:
        local = self._local_argv(model, prompt)
        if not self.ssh_host:
            return local
        # Run remotely; the remote shell needs one safely-quoted command string.
        inner = " ".join(shlex.quote(tok) for tok in local)
        if self.remote_shell:
            remote = f"{self.remote_shell} {shlex.quote(inner)}"
        else:
            remote = inner
        return ["ssh", self.ssh_host, "--", remote]

    def parse_result(self, stdout: str, stderr: str) -> dict:
        """Turn raw CLI output into fields. Override for a structured CLI.

        Returns a dict with at least ``text``; may include ``tokens_in``,
        ``tokens_out``, and ``meta``.
        """
        return {"text": stdout.strip()}

    def call(self, model: str, prompt: str) -> Completion:
        argv = self._argv(model, prompt)
        stdin = "" if self._uses_prompt_arg() else prompt
        started = time.monotonic()
        try:
            code, out, err = self._exec(argv, stdin, self.timeout_s)
        except subprocess.TimeoutExpired:
            raise ProviderError(f"{self.name} CLI timed out", retryable=True)
        except FileNotFoundError as exc:
            # CLI binary missing (or ssh missing). Non-retryable: reported n/a.
            raise ProviderError(f"{self.name} CLI not found: {exc}", retryable=False)
        latency = time.monotonic() - started
        if code != 0:
            retryable = code in (124, 137)  # timeout / killed
            raise ProviderError(
                f"{self.name} CLI exit {code}: {err.strip()[:300]}",
                retryable=retryable,
            )
        parsed = self.parse_result(out, err)
        meta = {"cli_startup_included": True}
        meta.update(parsed.get("meta", {}))
        return Completion(
            text=parsed.get("text", ""),
            latency_s=latency,
            tokens_in=parsed.get("tokens_in"),
            tokens_out=parsed.get("tokens_out"),
            raw_meta=meta,
        )
