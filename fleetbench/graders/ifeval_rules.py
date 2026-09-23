"""IFEval rule verifiers (ported subset).

Each item lists one or more rules; the score is the fraction of rules satisfied
and the item passes only if every rule holds. Verifiers here mirror the IFEval
instruction taxonomy (Google, Apache-2.0). Add verifiers as datasets need them;
an unknown rule raises so a gap is visible, never silently passed.
"""

from __future__ import annotations

import re


def _num_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _verify(rule: dict, response: str) -> bool:
    kind = rule["rule"]
    if kind == "keywords:existence":
        low = response.lower()
        return all(kw.lower() in low for kw in rule["keywords"])
    if kind == "punctuation:no_comma":
        return "," not in response
    if kind == "startend:end_checker":
        return response.strip().endswith(rule["end_phrase"])
    if kind == "length_constraints:number_words":
        n = _num_words(response)
        rel = rule["relation"]
        target = rule["num_words"]
        if rel == "at least":
            return n >= target
        if rel == "at most":
            return n <= target
        if rel == "exactly":
            return n == target
        raise ValueError(f"unknown length relation: {rel}")
    raise ValueError(f"unknown ifeval rule: {kind}")


def grade_ifeval(args: dict, response: str):
    rules = args["rules"]
    results = [(rule["rule"], _verify(rule, response)) for rule in rules]
    satisfied = sum(1 for _, ok in results if ok)
    total = len(results)
    score = satisfied / total if total else 0.0
    passed = satisfied == total
    return (score, passed, {"rules": results, "satisfied": satisfied, "total": total})
