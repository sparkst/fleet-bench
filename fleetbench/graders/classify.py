"""Classification graders: sanity (3-class) and TREC-6 (6-class).

Both extract a single label from the response by exact match first, then by the
earliest-occurring label mentioned. Grader functions return
``(score, passed, detail)``.
"""

from __future__ import annotations

import re
from typing import Optional


def _pick_label(response: str, labels: list[str], *, case_sensitive: bool) -> Optional[str]:
    text = response.strip()
    hay = text if case_sensitive else text.lower()
    # 1) exact match on the whole (stripped) response.
    for label in labels:
        cand = label if case_sensitive else label.lower()
        if hay == cand:
            return label
    # 2) earliest word-boundary occurrence.
    best: Optional[str] = None
    best_pos = len(hay) + 1
    for label in labels:
        cand = label if case_sensitive else label.lower()
        m = re.search(r"\b" + re.escape(cand) + r"\b", hay)
        if m and m.start() < best_pos:
            best_pos = m.start()
            best = label
    return best


def grade_classify(args: dict, response: str):
    labels = args["labels"]
    gold = args["answer"]
    picked = _pick_label(response, labels, case_sensitive=False)
    passed = picked is not None and picked.lower() == gold.lower()
    return (1.0 if passed else 0.0, passed, {"picked": picked, "gold": gold})


def grade_trec6(args: dict, response: str):
    labels = args["labels"]
    gold = args["answer"]
    # TREC labels are short uppercase codes; match case-insensitively but compare
    # against the canonical codes.
    picked = _pick_label(response, labels, case_sensitive=False)
    passed = picked is not None and picked.upper() == gold.upper()
    return (1.0 if passed else 0.0, passed, {"picked": picked, "gold": gold})
