"""Scrub redacts provider account/organization identifiers.

Synthetic ids are built from fragments so the id literal never appears verbatim
in this file (which the public-repo hygiene gate scans).
"""

from __future__ import annotations

from fleetbench.scrub import scrub

FAKE_ORG = "org_" + "abc1234567890def"
FAKE_ORG_DASH = "org-" + "ABC1234567890DEF"
FAKE_ACCT = "acct_" + "9988776655aa"
FAKE_ACCOUNT = "account_" + "abcdef123456"


def test_scrub_redacts_org_id():
    assert scrub(f"rate limit for {FAKE_ORG} tier") == "rate limit for org_[redacted] tier"


def test_scrub_redacts_openai_style_and_acct():
    assert "org_[redacted]" in scrub(f"{FAKE_ORG_DASH} exceeded")
    assert "acct_[redacted]" in scrub(f"{FAKE_ACCT} blocked")
    assert "account_[redacted]" in scrub(f"{FAKE_ACCOUNT} suspended")


def test_scrub_leaves_normal_text():
    assert scrub("HTTP 429 rate limit reached, try again in 2s") == (
        "HTTP 429 rate limit reached, try again in 2s"
    )
    assert scrub("") == ""


def test_scrub_redacts_home_dir_username():
    assert scrub("File /Users/alice/proj/x.py line 3") == "File /Users/[user]/proj/x.py line 3"
    assert scrub("at /home/bob/app.py") == "at /home/[user]/app.py"
