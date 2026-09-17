"""Graders: turn a model response into a score.

Deterministic graders (classify, gsm8k_numeric, trec6_label, ifeval_rules,
humaneval_exec) are dispatched by ``grade``. The judged suite ('judge') needs a
judge model and is handled by the runner (see ``fleetbench.graders.judge``), not
this dispatch.

A ``GradeResult`` carries a normalized score in [0, 1], an optional pass/fail,
and grader-specific detail for the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .classify import grade_classify, grade_trec6
from .humaneval_exec import grade_humaneval
from .ifeval_rules import grade_ifeval
from .numeric import grade_gsm8k


@dataclass
class GradeResult:
    score: float
    passed: Optional[bool] = None
    detail: dict = field(default_factory=dict)


_DETERMINISTIC = {
    "classify": grade_classify,
    "trec6_label": grade_trec6,
    "gsm8k_numeric": grade_gsm8k,
    "ifeval_rules": grade_ifeval,
    "humaneval_exec": grade_humaneval,
}


def is_deterministic(grader: str) -> bool:
    return grader in _DETERMINISTIC


def grade(grader: str, grader_args: dict, response: str) -> GradeResult:
    if grader not in _DETERMINISTIC:
        raise KeyError(f"grader '{grader}' is not a deterministic grader")
    score, passed, detail = _DETERMINISTIC[grader](grader_args, response)
    return GradeResult(score=score, passed=passed, detail=detail)


__all__ = ["GradeResult", "grade", "is_deterministic"]
