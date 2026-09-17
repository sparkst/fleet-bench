"""HumanEval grader: run the candidate code against hidden unit tests.

The model's response is stripped of markdown fences, concatenated with the
hidden ``check`` test, and executed in a SEPARATE process with a wall-clock
timeout. Pass = the process exits 0 (all asserts held).

Because this executes model-generated code:
- The subprocess runs with a SCRUBBED environment (no API keys or secrets), so a
  candidate cannot read live credentials from the environment.
- On POSIX it also gets a CPU-time rlimit (sized above the wall-clock timeout so
  a slow-but-correct solution is not falsely killed) and a memory rlimit.
- It runs in a throwaway temp directory.
This is process isolation, not a true sandbox: run untrusted models inside a
container (no network) for a stronger boundary.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from ..scrub import scrub

DEFAULT_TIMEOUT_S = 10.0

# Strip a fenced code block, tolerating any language tag (```py, ```python3, ...).
_FENCE_RE = re.compile(r"```[A-Za-z0-9_+.-]*[ \t]*\n?(.*?)```", re.DOTALL)


def _limit_preamble(cpu_seconds: int) -> str:
    return (
        "import resource\n"
        "try:\n"
        f"    resource.setrlimit(resource.RLIMIT_CPU, ({cpu_seconds}, {cpu_seconds}))\n"
        "    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))\n"
        "except Exception:\n"
        "    pass\n"
    )


def extract_code(response: str) -> str:
    """Return the Python code from a model response.

    Prefers fenced code blocks (joined); falls back to the raw response.
    """
    blocks = _FENCE_RE.findall(response)
    if blocks:
        return "\n\n".join(b.strip() for b in blocks)
    return response.strip()


def _build_program(code: str, test: str, entry_point: str, cpu_seconds: int) -> str:
    return (
        _limit_preamble(cpu_seconds)
        + "\n"
        + code
        + "\n\n"
        + test
        + f"\n\ncheck({entry_point})\n"
    )


def _scrubbed_env(tmp: str) -> dict:
    # Minimal environment: no API keys or secrets reach the candidate process.
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": tmp,
        "TMPDIR": tmp,
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _run_program(program: str, timeout_s: float) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        prog_path = Path(tmp) / "candidate.py"
        prog_path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(prog_path)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=tmp,
                env=_scrubbed_env(tmp),
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if proc.returncode == 0:
            return True, "ok"
        # Redact the throwaway temp dir and any home path from the traceback so a
        # failure reason committed to a public artifact carries no local path.
        reason = (proc.stderr.strip()[-300:] or f"exit {proc.returncode}")
        return False, scrub(reason.replace(tmp, "<tmpdir>"))


def grade_humaneval(args: dict, response: str):
    code = extract_code(response)
    entry_point = args["entry_point"]
    test = args["test"]
    timeout_s = float(args.get("timeout_s", DEFAULT_TIMEOUT_S))
    # CPU ceiling comfortably above the wall-clock timeout so a legitimately slow
    # solution is not killed by SIGXCPU before the intended timeout.
    cpu_seconds = int(timeout_s) + 5
    program = _build_program(code, test, entry_point, cpu_seconds)
    passed, reason = _run_program(program, timeout_s)
    return (1.0 if passed else 0.0, passed, {"reason": reason, "entry_point": entry_point})
