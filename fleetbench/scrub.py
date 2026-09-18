"""Redact provider-controlled identifiers from text before it is persisted.

Upstream error bodies (for example a 429 from a provider) can embed an account
or organization identifier. Those must never reach a committed public artifact,
so error text is scrubbed at the source before it becomes an n/a reason or a
report cell.
"""

from __future__ import annotations

import re

# Account/organization identifier shapes: OpenAI-style `org-...`, Groq-style
# `org_...`, and similar `account_`/`acct_` slugs. Also home-directory prefixes,
# which carry a local username, redacted so error strings do not leak them.
# Require a digit in the id token so real ids (org_01kh..., acct_99...) are
# redacted while plain hyphenated words (account-summary) are left intact.
_PATTERNS = [
    (re.compile(r"\borg[_-](?=[A-Za-z0-9]*\d)[A-Za-z0-9]{6,}\b"), "org_[redacted]"),
    (re.compile(r"\bacct[_-](?=[A-Za-z0-9]*\d)[A-Za-z0-9]{6,}\b"), "acct_[redacted]"),
    (re.compile(r"\baccount[_-](?=[A-Za-z0-9]*\d)[A-Za-z0-9]{6,}\b"), "account_[redacted]"),
    (re.compile(r"/Users/[^/\s]+"), "/Users/[user]"),
    (re.compile(r"/home/[^/\s]+"), "/home/[user]"),
    # Defensive: Cloudflare account/zone ids are bare 32-char lowercase-hex
    # tokens with no prefix, interpolated straight into the base_url. If CF
    # ever echoes the URL/account id in an error body it would otherwise pass
    # through unredacted. This also matches other 32-hex tokens (e.g. a git
    # sha is 40 chars so is unaffected, but a truncated 32-char hex string in
    # normal text would also be redacted) - accepted tradeoff for safety
    # since a bare 32-hex string is not expected in ordinary error prose.
    (re.compile(r"\b[0-9a-f]{32}\b"), "[redacted-hex]"),
]


def scrub(text: str) -> str:
    """Redact known account/organization identifiers from ``text``."""
    if not text:
        return text
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text
