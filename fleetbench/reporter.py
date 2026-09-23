"""Render one markdown report per run from JSONL rows plus the manifest.

ASCII only, no em dashes (REQ-FB-06). Tables are model x metric per dataset,
latency medians and p90, burn per provider (requests and tokens where the
provider reports them), n/a counts by reason, and an appendix with the exact
prompts.
"""

from __future__ import annotations

import math
from collections import defaultdict


def _median(values: list[float]):
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _percentile(values: list[float], pct: float):
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    # Nearest-rank method: rank = ceil(p/100 * n) (ceil, not banker's round).
    rank = max(1, min(len(s), math.ceil(pct / 100.0 * len(s))))
    return s[rank - 1]


def _fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _scored_rows(rows):
    return [r for r in rows if not r.get("na_reason")]


# Suite names that are non-discriminative smoke runs, not model-ranking runs.
_SMOKE_SUITES = {"phase1", "smoke"}

_SMOKE_CAVEAT = (
    "> NOTE: This is a smoke run (3 items per dataset, 1 per SDLC task type). "
    "The near-universal 1.000 scores reflect trivial item difficulty, not "
    "model quality parity. Do NOT read these tables as a model ranking. See "
    "the Phase 2 report for discriminative results."
)


def render(rows: list[dict], manifest: dict) -> str:
    out: list[str] = []
    run_id = manifest.get("run_id", "run")
    out.append(f"# fleet-bench report: {run_id}")
    out.append("")
    if manifest.get("suite") in _SMOKE_SUITES:
        out.append(_SMOKE_CAVEAT)
        out.append("")
    out.append(f"- date: {manifest.get('date')}")
    out.append(f"- suite: {manifest.get('suite')}")
    out.append(f"- git sha: {manifest.get('git_sha')}")
    out.append(f"- hosts in this report: {', '.join(sorted({r['host'] for r in rows})) or 'n/a'}")
    out.append(f"- seed: {manifest.get('seed')}")
    out.append(f"- started: {manifest.get('started')}  ended: {manifest.get('ended')}")
    out.append("")
    out.append("Latency includes CLI startup for the CLI-backed providers "
               "(anthropic, codex, copilot).")
    out.append("")

    out.append("## Datasets and licenses")
    out.append("")
    out.append("| dataset | items | license | provenance |")
    out.append("|---|---|---|---|")
    for name, meta in manifest.get("datasets", {}).items():
        prov = (meta.get("provenance") or "").replace("|", " ").replace("\n", " ")
        out.append(f"| {name} | {meta.get('count')} | {meta.get('license','')} | {prov} |")
    out.append("")

    out.extend(_accuracy_tables(rows))
    out.extend(_latency_table(rows))
    out.extend(_burn_table(rows, manifest))
    out.extend(_na_table(rows))
    out.extend(_judge_section(manifest))
    out.extend(_prompt_appendix(rows))
    return "\n".join(out) + "\n"


def _accuracy_tables(rows) -> list[str]:
    out = ["## Accuracy by dataset (model x metric)", ""]
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)
    for dataset in sorted(by_ds):
        drows = by_ds[dataset]
        models = sorted({r["model"] for r in drows})
        judged = dataset == "sdlc"
        metric = "median judge score (0-1)" if judged else "pass rate"
        out.append(f"### {dataset}")
        out.append("")
        out.append(f"| model | {metric} | mean score | n | n/a |")
        out.append("|---|---|---|---|---|")
        for model in models:
            mrows = [r for r in drows if r["model"] == model]
            scored = _scored_rows(mrows)
            na = len(mrows) - len(scored)
            scores = [r["score"] for r in scored if r.get("score") is not None]
            if judged:
                primary = _median(scores) if scores else None
            else:
                passes = [1 if r.get("passed") else 0 for r in scored if r.get("passed") is not None]
                primary = (sum(passes) / len(passes)) if passes else None
            mean_score = (sum(scores) / len(scores)) if scores else None
            out.append(
                f"| {model} | {_fmt(primary)} | {_fmt(mean_score)} | {len(scored)} | {na} |"
            )
        out.append("")
    return out


def _latency_table(rows) -> list[str]:
    out = ["## Latency by model (seconds)", "",
           "| provider | model | median | p90 | n |", "|---|---|---|---|---|"]
    by_model = defaultdict(list)
    for r in _scored_rows(rows):
        by_model[(r["provider"], r["model"])].append(r["latency_s"])
    for (provider, model) in sorted(by_model):
        lats = by_model[(provider, model)]
        out.append(
            f"| {provider} | {model} | {_fmt(_median(lats))} | {_fmt(_percentile(lats, 90))} | {len(lats)} |"
        )
    out.append("")
    return out


def _burn_table(rows, manifest) -> list[str]:
    out = ["## Burn per provider", "",
           "| provider | requests | tokens_in | tokens_out | cap (calls) |",
           "|---|---|---|---|---|"]
    by_prov = defaultdict(lambda: {"req": 0, "tin": 0, "tout": 0, "has_tokens": False})
    for r in rows:
        if r.get("na_reason"):
            continue
        agg = by_prov[r["provider"]]
        agg["req"] += 1
        if r.get("tokens_in") is not None:
            agg["tin"] += r["tokens_in"]
            agg["has_tokens"] = True
        if r.get("tokens_out") is not None:
            agg["tout"] += r["tokens_out"]
            agg["has_tokens"] = True
    caps = manifest.get("caps", {}).get("providers", {})
    for provider in sorted(by_prov):
        agg = by_prov[provider]
        cap = caps.get(provider, {}).get("max_calls", 0)
        cap_str = "unlimited" if not cap else str(cap)
        tin = agg["tin"] if agg["has_tokens"] else "n/a"
        tout = agg["tout"] if agg["has_tokens"] else "n/a"
        out.append(f"| {provider} | {agg['req']} | {tin} | {tout} | {cap_str} |")
    out.append("")
    return out


def _na_table(rows) -> list[str]:
    out = ["## Not-available (reported, never silently skipped)", ""]
    na = [r for r in rows if r.get("na_reason")]
    if not na:
        out.append("None.")
        out.append("")
        return out
    counts = defaultdict(lambda: defaultdict(int))
    for r in na:
        counts[(r["provider"], r["model"])][r["na_reason"]] += 1
    out.append("| provider | model | reason | count |")
    out.append("|---|---|---|---|")
    for (provider, model) in sorted(counts):
        for reason, c in sorted(counts[(provider, model)].items()):
            safe = (reason or "").replace("|", " ").replace("\n", " ").replace("\r", " ")
            out.append(f"| {provider} | {model} | {safe} | {c} |")
    out.append("")
    return out


def _judge_section(manifest) -> list[str]:
    j = manifest.get("judge") or {}
    if not j:
        return []
    out = ["## Judge", ""]
    out.append(f"- items judged: {j.get('judged', 0)}")
    cal = j.get("calibration") or {}
    if cal.get("enabled"):
        out.append(
            f"- calibration re-judge by {cal.get('calibration_model')}: "
            f"n={cal.get('n')}, mean abs score diff={_fmt(cal.get('mean_abs_diff'))}"
        )
    else:
        out.append("- calibration: not run")
    out.append("")
    return out


def _prompt_appendix(rows) -> list[str]:
    out = ["## Appendix: exact prompts", ""]
    seen = {}
    for r in rows:
        key = (r["dataset"], r["item_id"])
        if key not in seen:
            seen[key] = r["prompt"]
    for (dataset, item_id) in sorted(seen):
        out.append(f"### {dataset} / {item_id}")
        out.append("")
        out.append("```")
        out.append(seen[(dataset, item_id)])
        out.append("```")
        out.append("")
    return out
