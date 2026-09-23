"""Classification graders: sanity (3-class) and TREC-6 (6-class).

Both extract a single label from the response by exact match first, then by
the label occurrence nearest the end of the response (the "final answer"
convention), skipping any occurrence that is negated. Grader functions
return ``(score, passed, detail)``.
"""

from __future__ import annotations

import re
from typing import Optional

_NEGATION_TOKENS = (
    "not",
    "isn't",
    "isnt",
    "is not",
    "no",
    "never",
    "rather than",
)

# Words-of-lookback to scan for a negation token before a candidate match.
_NEGATION_WINDOW_WORDS = 3


def _is_negated(hay: str, match_start: int) -> bool:
    """Return True if a negation token appears within ~3 words before the
    match at ``match_start`` in ``hay``."""
    prefix = hay[:match_start]
    words = re.findall(r"[a-z']+", prefix)
    window = words[-_NEGATION_WINDOW_WORDS:]
    window_text = " ".join(window)
    for neg in _NEGATION_TOKENS:
        if neg in window_text:
            return True
    return False


def _pick_label(
    response: str, labels: list[str], *, case_sensitive: bool
) -> tuple[Optional[str], str]:
    """Pick the label to score against gold.

    Returns ``(label, confidence)`` where confidence is "high" for the
    normal final-answer pick (or exact match) and "low" when we had to fall
    back to the earliest-occurring label because every candidate was
    negated or nothing survived the negation filter.
    """
    text = response.strip()
    hay = text if case_sensitive else text.lower()
    # 1) exact match on the whole (stripped) response.
    for label in labels:
        cand = label if case_sensitive else label.lower()
        if hay == cand:
            return label, "high"

    # 2) collect all word-boundary occurrences of every label.
    occurrences = []  # (start, label)
    for label in labels:
        cand = label if case_sensitive else label.lower()
        for m in re.finditer(r"\b" + re.escape(cand) + r"\b", hay):
            occurrences.append((m.start(), label))

    if not occurrences:
        return None, "high"

    # 3) prefer the occurrence nearest the end that is not negated.
    non_negated = [
        (pos, label) for pos, label in occurrences if not _is_negated(hay, pos)
    ]
    if non_negated:
        non_negated.sort(key=lambda item: item[0])
        return non_negated[-1][1], "high"

    # 4) nothing survived the negation filter: fall back to earliest match,
    # flagged as low confidence.
    occurrences.sort(key=lambda item: item[0])
    return occurrences[0][1], "low"


def grade_classify(args: dict, response: str):
    labels = args["labels"]
    gold = args["answer"]
    picked, confidence = _pick_label(response, labels, case_sensitive=False)
    passed = picked is not None and picked.lower() == gold.lower()
    detail = {"picked": picked, "gold": gold}
    if confidence == "low":
        detail["picked_confidence"] = "low"
    return (1.0 if passed else 0.0, passed, detail)


def grade_trec6(args: dict, response: str):
    labels = args["labels"]
    gold = args["answer"]
    # TREC labels are short uppercase codes; match case-insensitively but compare
    # against the canonical codes.
    picked, confidence = _pick_label(response, labels, case_sensitive=False)
    passed = picked is not None and picked.upper() == gold.upper()
    detail = {"picked": picked, "gold": gold}
    if confidence == "low":
        detail["picked_confidence"] = "low"
    return (1.0 if passed else 0.0, passed, detail)
