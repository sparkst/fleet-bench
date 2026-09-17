"""GSM8K numeric grader.

Extracts the final integer answer: prefer the number after the last '####'
marker (the GSM8K convention we ask the model to emit), else the last integer
anywhere in the response. Compares as integers.
"""

from __future__ import annotations

import re


def _extract_answer(response: str):
    """Return (value, marker_present). marker_present is False when we fall back
    to the last integer, so a grade carries that caveat in its detail."""
    # Prefer the '#### <number>' marker.
    marker = re.findall(r"####\s*(-?\d[\d,]*)", response)
    if marker:
        return _to_int(marker[-1]), True
    # Fallback: the last integer in the text (heuristic).
    nums = re.findall(r"-?\d[\d,]*", response)
    if nums:
        return _to_int(nums[-1]), False
    return None, False


def _to_int(token: str):
    try:
        return int(token.replace(",", ""))
    except ValueError:
        return None


def grade_gsm8k(args: dict, response: str):
    gold = int(str(args["answer"]).replace(",", ""))
    got, marker_present = _extract_answer(response)
    passed = got is not None and got == gold
    return (
        1.0 if passed else 0.0,
        passed,
        {"extracted": got, "gold": gold, "marker_present": marker_present},
    )
