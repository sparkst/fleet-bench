"""Public-repo hygiene: no secrets, no internal identifiers, ASCII only.

Scans tracked plus untracked-not-ignored files. This is the secret-pattern gate
the public repo relies on in CI (running gitleaks locally is a recommended extra,
not a CI job). The regexes require a realistic key/id suffix, so this test's own
source does not match them.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from fleetbench.providers.hetzner import HetznerProvider

REPO_ROOT = Path(__file__).resolve().parents[1]

# Secret shapes. Each requires a long realistic suffix so the pattern literal in
# this file cannot match itself.
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{24,}"),          # OpenAI-style
    re.compile(r"sk-or-v1-[0-9a-f]{24,}"),        # OpenRouter
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),    # GitHub tokens
    re.compile(r"github_pat_[A-Za-z0-9_]{50,}"),  # fine-grained PAT
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}"),  # Slack
    re.compile(r"AKIA[0-9A-Z]{16}"),              # AWS access key id
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"gsk_[A-Za-z0-9]{40,}"),          # Groq key shape
    # Provider account/organization identifiers (Groq org_..., OpenAI org-...).
    # Aligned with scrub.py: {6,} with a required digit so a real id is caught
    # while plain hyphenated words (account-summary) are not false positives.
    re.compile(r"\borg[_-](?=[A-Za-z0-9]*\d)[A-Za-z0-9]{6,}\b"),
    re.compile(r"\bacct[_-](?=[A-Za-z0-9]*\d)[A-Za-z0-9]{6,}\b"),
    re.compile(r"\baccount[_-](?=[A-Za-z0-9]*\d)[A-Za-z0-9]{6,}\b"),
]

# Internal identifiers that must never appear in the public repo. Built from
# fragments so the token literal does not appear verbatim in this scanner.
INTERNAL_TOKENS = [
    ".ts" + ".net",           # tailnet host suffix
    "ancon" + "-cliff",       # tailnet name
    "192." + "168.",          # private LAN range
    "hooks.sparkry" + ".ai",  # internal fleet endpoint
]

EM_DASH = chr(0x2014)
EN_DASH = chr(0x2013)

# Binary/asset extensions to skip when reading as text.
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip"}


def _tracked_files() -> list[Path]:
    # Tracked plus untracked-but-not-ignored, so the gate covers new files
    # before they are committed. Ignored paths (.venv, caches) are excluded.
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    seen = []
    for line in out.stdout.splitlines():
        if line.strip():
            seen.append(REPO_ROOT / line)
    return seen


def _text_files() -> list[Path]:
    return [
        p for p in _tracked_files()
        if p.suffix.lower() not in SKIP_SUFFIXES and p.exists()
    ]


def test_no_secret_patterns():
    offenders = []
    for path in _text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pat in SECRET_PATTERNS:
            if pat.search(text):
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {pat.pattern}")
    assert not offenders, f"possible secrets found: {offenders}"


def test_no_internal_identifiers():
    offenders = []
    for path in _text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in INTERNAL_TOKENS:
            if token in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {token}")
    assert not offenders, f"internal identifiers found: {offenders}"


def test_md_and_py_are_ascii_no_dashes():
    # The ASCII/no-dash house style applies to human-authored prose and config.
    # Raw model responses in results/*.jsonl are DATA, not prose, and a model may
    # legitimately emit non-ASCII; those are deliberately exempt from this check
    # (they are still covered by the secret and internal-identifier scans above).
    offenders = []
    for path in _text_files():
        if path.suffix.lower() not in {".md", ".py", ".yaml", ".yml", ".json", ".toml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if EM_DASH in text or EN_DASH in text:
            offenders.append(f"{path.relative_to(REPO_ROOT)}: dash")
        try:
            text.encode("ascii")
        except UnicodeEncodeError:
            offenders.append(f"{path.relative_to(REPO_ROOT)}: non-ascii")
    assert not offenders, f"style violations: {offenders}"


def test_org_id_pattern_would_catch_a_leak():
    # Regression: the class of leak that slipped through once (a provider org id in
    # a committed report) must now be caught by the secret-pattern gate. The
    # synthetic id is built from fragments so it does not appear verbatim here.
    synthetic = "error for " + "org_" + "abc1234567890def" + " service tier"
    assert any(p.search(synthetic) for p in SECRET_PATTERNS)


def test_hetzner_ledger_default_path_is_gitignored():
    # The ledger holds run-accounting data and must never be committed. Assert the
    # default path from the provider is covered by an active .gitignore pattern, so
    # renaming either side is caught.
    default_path = HetznerProvider().ledger_path
    result = subprocess.run(
        ["git", "check-ignore", default_path],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"ledger path not gitignored: {default_path}"
