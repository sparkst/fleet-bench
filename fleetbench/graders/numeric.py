"""GSM8K numeric grader.

Extracts the final integer answer: prefer the number after the last '####'
marker (the GSM8K convention we ask the model to emit). When that marker is
absent, we do NOT scan the whole response (an intermediate-step number
matching gold by coincidence would false-positive); instead we restrict
extraction to the last non-empty line, or an explicit 'Answer: <n>' if
present. If no integer can be found there, the grade is n/a (not a
pass/fail). Compares as integers.
"""

from __future__ import annotations

import re


def _extract_answer(response: str):
    """Return (value, marker_present). marker_present is False when we fall
    back to the no-marker heuristic, so a grade carries that caveat in its
    detail."""
    # Prefer the '#### <number>' marker.
    marker = re.findall(r"####\s*(-?\d[\d,]*)", response)
    if marker:
        return _to_int(marker[-1]), True

    # No marker: look for an explicit 'Answer: <n>' anywhere (last one wins).
    explicit = re.findall(r"(?i)answer:\s*(-?\d[\d,]*)", response)
    if explicit:
        return _to_int(explicit[-1]), False

    # Otherwise restrict extraction to the last non-empty line, and within
    # that line to the last sentence/clause (split on '.'), so trailing
    # prose that mentions an intermediate number earlier in the same line
    # cannot masquerade as the final answer.
    lines = [line for line in response.splitlines() if line.strip()]
    if lines:
        segments = [seg for seg in lines[-1].split(".") if seg.strip()]
        if segments:
            nums = re.findall(r"-?\d[\d,]*", segments[-1])
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
    if got is None:
        return (
            None,
            None,
            {
                "extracted": None,
                "gold": gold,
                "marker_present": marker_present,
                "na_reason": "no marker and no integer on the final line",
            },
        )
    passed = got == gold
    detail = {"extracted": got, "gold": gold, "marker_present": marker_present}
    if not marker_present:
        detail["picked_confidence"] = "low"
    return (1.0 if passed else 0.0, passed, detail)
