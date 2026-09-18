"""Dataset loaders.

Phase 1 reads vendored JSON fixtures under ``data/phase1/`` so unit tests and CI
run with no network. Each loader normalizes its fixture into ``Item`` records and
builds the exact prompt the model sees. Ordering in the fixture is the pinned
order; ``load(name, n)`` takes the first ``n`` (deterministic, no random sampling
in Phase 1). Phase 2 loaders (full datasets with pinned ids) plug in behind the
same ``Item`` contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "phase1"

DATASET_NAMES = ["ifeval", "humaneval", "trec6", "gsm8k", "sanity", "sdlc"]


@dataclass
class Item:
    id: str
    dataset: str
    prompt: str
    grader: str
    grader_args: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    # Relative cost weight of this item; a provider whose caps set
    # skip_multiplier_gt below this value skips the item (budget protection).
    # Default 1.0; a fixture may raise it for expensive items.
    cost_multiplier: float = 1.0


def _read_fixture(name: str) -> dict:
    path = DATA_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"no fixture for dataset '{name}' at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# --- per-dataset prompt builders ---------------------------------------------
def _build_sanity(fx: dict) -> list[Item]:
    labels = fx["labels"]
    label_str = ", ".join(labels)
    items = []
    for it in fx["items"]:
        prompt = (
            f"Classify the following item as exactly one of: {label_str}. "
            f"Respond with only the category and nothing else.\n\nItem: {it['word']}"
        )
        items.append(
            Item(
                id=it["id"],
                dataset="sanity",
                prompt=prompt,
                grader="classify",
                grader_args={"answer": it["answer"], "labels": labels},
            )
        )
    return items


def _build_gsm8k(fx: dict) -> list[Item]:
    items = []
    for it in fx["items"]:
        prompt = (
            "Solve the following grade-school math problem. Show brief reasoning, "
            "then end your response with a line of the form '#### <integer>'.\n\n"
            + it["question"]
        )
        items.append(
            Item(
                id=it["id"],
                dataset="gsm8k",
                prompt=prompt,
                grader="gsm8k_numeric",
                grader_args={"answer": it["answer"]},
            )
        )
    return items


def _build_trec6(fx: dict) -> list[Item]:
    labels = fx["labels"]
    help_text = fx.get("label_help", "")
    items = []
    for it in fx["items"]:
        prompt = (
            "Classify the question into exactly one of these six categories: "
            f"{', '.join(labels)} ({help_text}). "
            "Respond with only the category code.\n\nQuestion: " + it["question"]
        )
        items.append(
            Item(
                id=it["id"],
                dataset="trec6",
                prompt=prompt,
                grader="trec6_label",
                grader_args={"answer": it["answer"], "labels": labels},
            )
        )
    return items


def _build_ifeval(fx: dict) -> list[Item]:
    items = []
    for it in fx["items"]:
        items.append(
            Item(
                id=it["id"],
                dataset="ifeval",
                prompt=it["prompt"],
                grader="ifeval_rules",
                grader_args={"rules": it["rules"]},
            )
        )
    return items


def _build_humaneval(fx: dict) -> list[Item]:
    items = []
    for it in fx["items"]:
        prompt = (
            "Complete the following Python function. Return only the full function "
            "definition as Python code, no explanation.\n\n" + it["prompt"]
        )
        items.append(
            Item(
                id=it["id"],
                dataset="humaneval",
                prompt=prompt,
                grader="humaneval_exec",
                grader_args={
                    "entry_point": it["entry_point"],
                    "test": it["test"],
                    "prompt": it["prompt"],
                },
            )
        )
    return items


def _build_sdlc(fx: dict) -> list[Item]:
    items = []
    for it in fx["items"]:
        items.append(
            Item(
                id=it["id"],
                dataset="sdlc",
                prompt=it["prompt"],
                grader="judge",
                grader_args={"rubric": it["rubric"]},
                meta={"task_type": it["task_type"], "rubric": it["rubric"]},
            )
        )
    return items


_BUILDERS = {
    "sanity": _build_sanity,
    "gsm8k": _build_gsm8k,
    "trec6": _build_trec6,
    "ifeval": _build_ifeval,
    "humaneval": _build_humaneval,
    "sdlc": _build_sdlc,
}


def load(name: str, n: Optional[int] = None) -> list[Item]:
    """Load dataset ``name``; take the first ``n`` items (pinned order).

    Raises ``ValueError`` if ``n`` exceeds the number of vendored fixture
    items: a config that requests a full-size Phase 2 count against the
    Phase-1-sized fixtures must fail loudly, not silently truncate or
    misbehave (REQ-FB-03).
    """
    if name not in _BUILDERS:
        raise KeyError(f"unknown dataset: {name}")
    fx = _read_fixture(name)
    items = _BUILDERS[name](fx)
    available = len(items)
    if n is not None and n > available:
        raise ValueError(
            f"dataset '{name}' requests {n} items but only {available} vendored "
            "fixtures are available; the full-size loader has not landed yet"
        )
    return items[:n] if n is not None else items


def fixture_meta(name: str) -> dict:
    """License and provenance for a dataset, for the report and manifest."""
    fx = _read_fixture(name)
    return {
        "dataset": name,
        "license": fx.get("license", ""),
        "provenance": fx.get("provenance", ""),
        "count": len(fx.get("items", [])),
    }
