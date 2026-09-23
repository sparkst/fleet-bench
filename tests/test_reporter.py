"""Reporter rendering: tables, n/a section, ASCII-only output."""

from __future__ import annotations

from fleetbench.reporter import _percentile, render

ROWS = [
    {"suite": "t", "host": "primary", "provider": "groq", "model": "m1",
     "dataset": "sanity", "item_id": "sanity-000", "prompt": "classify apple",
     "response": "fruit", "latency_s": 0.5, "tokens_in": 10, "tokens_out": 2,
     "attempts": 1, "na_reason": None, "grader": "classify", "task_type": None,
     "score": 1.0, "passed": True},
    {"suite": "t", "host": "primary", "provider": "groq", "model": "m1",
     "dataset": "sanity", "item_id": "sanity-001", "prompt": "classify sofa",
     "response": "fruit", "latency_s": 0.7, "tokens_in": 10, "tokens_out": 2,
     "attempts": 1, "na_reason": None, "grader": "classify", "task_type": None,
     "score": 0.0, "passed": False},
    {"suite": "t", "host": "primary", "provider": "codex", "model": "gpt-5.6-sol",
     "dataset": "sanity", "item_id": "sanity-000", "prompt": "classify apple",
     "response": "", "latency_s": 0.0, "tokens_in": None, "tokens_out": None,
     "attempts": 0, "na_reason": "model-unavailable-on-plan", "grader": "classify",
     "task_type": None, "score": None, "passed": None},
]

MANIFEST = {
    "run_id": "2026-01-01-t", "date": "2026-01-01", "suite": "t",
    "git_sha": "abc123", "seed": 0, "started": "s", "ended": "e",
    "datasets": {"sanity": {"count": 2, "license": "MIT", "provenance": "authored"}},
    "caps": {"providers": {"groq": {"max_calls": 0}, "codex": {"max_calls": 200}}},
    "judge": {"judged": 0},
}


def test_render_has_sections_and_stats():
    md = render(ROWS, MANIFEST)
    assert "# fleet-bench report: 2026-01-01-t" in md
    assert "## Accuracy by dataset" in md
    assert "## Latency by model" in md
    assert "## Burn per provider" in md
    assert "## Not-available" in md
    assert "## Appendix: exact prompts" in md
    # sanity pass rate for m1 is 0.5 (1 of 2).
    assert "0.500" in md
    # n/a reason surfaced.
    assert "model-unavailable-on-plan" in md
    # exact prompt present in appendix.
    assert "classify apple" in md


def test_report_is_ascii_and_no_em_dash():
    md = render(ROWS, MANIFEST)
    md.encode("ascii")  # raises if any non-ASCII slipped in
    assert chr(0x2014) not in md  # em dash
    assert chr(0x2013) not in md  # en dash


def test_percentile_uses_ceil_nearest_rank():
    assert _percentile([1, 2, 3, 4, 5], 90) == 5
    assert _percentile(list(range(1, 11)), 90) == 9
    assert _percentile([42], 90) == 42


def test_na_reason_is_sanitized_in_table():
    rows = [dict(ROWS[2], na_reason="boom | with | pipes\nand newline")]
    md = render(rows, MANIFEST)
    # The reason must not inject extra table columns or break onto a new line.
    line = [ln for ln in md.splitlines() if "boom" in ln][0]
    assert "pipes and newline" in line  # newline collapsed to a space
    assert "boom   with   pipes" in line  # pipes replaced by spaces
