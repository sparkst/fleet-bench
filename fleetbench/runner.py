"""The benchmark runner.

One rate-limited worker per provider, providers running concurrently; within a
provider, up to ``concurrency`` items run in parallel (from caps.yaml). Each call
uses the provider base's backoff and n/a-after-3 rule. Deterministic datasets are
graded inline; the SDLC suite is graded afterward by a blind judge pass
(randomized processing order, hidden model ids, a calibration re-judge by a second
judge model).

Host sharding (REQ-FB-05): each provider is assigned a host label; a runner only
executes providers whose host matches its own and writes host-scoped output
(``results-<host>.jsonl`` + ``manifest-<host>.json``). ``merge`` combines the row
files and ``merge_manifests`` unions the per-host manifests at report time.

Nothing is silently skipped: a model not on a plan, a missing key, a cap hit, a
cost-multiplier skip, a provider construction failure, or a judge failure all
produce a recorded row with an ``na_reason``.
"""

from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import datasets as ds
from .caps import CapCounter, load_caps
from .graders import grade, is_deterministic
from .graders.judge import (
    build_judge_prompt,
    parse_judge_response,
    recognized_dimension_count,
    score_from_dimensions,
)
from .scrub import scrub
from .providers import build_provider


@dataclass
class RunConfig:
    suite: str
    date: str
    host: str
    datasets: dict  # name -> count
    providers: list  # list of {name, host, models?, overrides?}
    judge: dict = field(default_factory=dict)
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        return cls(
            suite=d["suite"],
            date=d["date"],
            host=d["host"],
            datasets=d["datasets"],
            providers=d["providers"],
            judge=d.get("judge", {}),
            seed=int(d.get("seed", 0)),
        )


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Runner:
    def __init__(self, config: RunConfig, caps_path: Optional[str] = None, results_root: str = "results"):
        self.config = config
        self.caps = load_caps(caps_path)
        self.counter = CapCounter(self.caps)
        self.results_root = Path(results_root)
        self.rows: list[dict] = []
        self._items_by_dataset = {
            name: ds.load(name, n) for name, n in config.datasets.items()
        }

    # --- selection ------------------------------------------------------------
    def _local_providers(self) -> list[dict]:
        return [p for p in self.config.providers if p.get("host", self.config.host) == self.config.host]

    def _all_items(self) -> list:
        items = []
        for name in self.config.datasets:
            items.extend(self._items_by_dataset[name])
        return items

    def _provider_overrides(self, name: str, pconf: dict) -> dict:
        """rpm and concurrency come from caps.yaml unless the run-config overrides."""
        overrides = dict(pconf.get("overrides") or {})
        overrides.setdefault("rpm", self.counter.rpm_for(name))
        overrides.setdefault("concurrency", self.counter.concurrency_for(name))
        return overrides

    # --- one provider's work --------------------------------------------------
    def _run_provider(self, pconf: dict) -> list[dict]:
        name = pconf["name"]
        try:
            provider = build_provider(name, self._provider_overrides(name, pconf))
        except Exception as exc:  # construction failure: report, do not abort the run
            models = pconf.get("models") or [name]
            return [
                self._na_row(name, model, item, f"provider-error: {type(exc).__name__}: {exc}")
                for model in models
                for item in self._all_items()
            ]

        requested = pconf.get("models") or provider.available_models()
        available = set(provider.available_models())
        skip_gt = self.caps.get("providers", {}).get(name, {}).get("skip_multiplier_gt")
        concurrency = max(1, int(getattr(provider, "concurrency", 1) or 1))

        tasks = [(model, item) for model in requested for item in self._all_items()]

        def work(model_item):
            model, item = model_item
            if model not in available:
                return self._na_row(name, model, item, "model-unavailable-on-plan")
            if skip_gt is not None and item.cost_multiplier > float(skip_gt):
                return self._na_row(name, model, item, "cost-multiplier-exceeded")
            if not self.counter.try_reserve(name):
                return self._na_row(name, model, item, "cap-exceeded")
            completion = provider.run_item(model, item.prompt)
            return self._row(name, model, item, completion)

        if concurrency == 1:
            return [work(t) for t in tasks]
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            return list(pool.map(work, tasks))

    def _row(self, provider: str, model: str, item, completion) -> dict:
        row = {
            "suite": self.config.suite,
            "host": self.config.host,
            "provider": provider,
            "model": model,
            "dataset": item.dataset,
            "item_id": item.id,
            "prompt": item.prompt,
            "response": completion.text,
            "latency_s": round(completion.latency_s, 4),
            "tokens_in": completion.tokens_in,
            "tokens_out": completion.tokens_out,
            "attempts": completion.attempts,
            "na_reason": completion.na_reason,
            "grader": item.grader,
            "task_type": item.meta.get("task_type"),
            "ts": _now(),
        }
        if completion.is_na:
            row["score"] = None
            row["passed"] = None
            return row
        if is_deterministic(item.grader):
            result = grade(item.grader, item.grader_args, completion.text)
            row["score"] = result.score
            row["passed"] = result.passed
            row["grade_detail"] = result.detail
        else:
            # Judged suite: scored later.
            row["score"] = None
            row["passed"] = None
            row["_rubric"] = item.meta.get("rubric")
        return row

    def _na_row(self, provider: str, model: str, item, reason: str) -> dict:
        return {
            "suite": self.config.suite,
            "host": self.config.host,
            "provider": provider,
            "model": model,
            "dataset": item.dataset,
            "item_id": item.id,
            "prompt": item.prompt,
            "response": "",
            "latency_s": 0.0,
            "tokens_in": None,
            "tokens_out": None,
            "attempts": 0,
            "na_reason": reason,
            "grader": item.grader,
            "task_type": item.meta.get("task_type"),
            "score": None,
            "passed": None,
            "ts": _now(),
        }

    # --- judge pass -----------------------------------------------------------
    def _judge_pass(self, rows: list[dict]) -> dict:
        jconf = self.config.judge
        judged = [r for r in rows if r["grader"] == "judge" and not r["na_reason"]]
        if not jconf:
            # No judge configured: judged-suite rows are not evaluated. Mark them
            # n/a rather than leaving score=None with na_reason=None (which the
            # reporter would count as "scored" with an n/a metric).
            for r in judged:
                r["na_reason"] = "judge-not-configured"
                r.pop("_rubric", None)
            return {"judged": 0}
        if not judged:
            return {"judged": 0}
        import random

        rng = random.Random(self.config.seed)
        order = list(range(len(judged)))
        rng.shuffle(order)  # processing order only; each item is judged in isolation

        judge = build_provider(jconf["provider"], jconf.get("overrides"))
        judge_model = jconf["model"]
        judged_ok = 0
        for idx in order:
            r = judged[idx]
            if not self.counter.try_reserve("judge"):
                r["na_reason"] = "judge-cap-exceeded"
                continue
            prompt = build_judge_prompt(
                r.get("task_type") or "task", r["prompt"], r["_rubric"], r["response"]
            )
            completion = judge.run_item(judge_model, prompt)
            if completion.is_na or not completion.text.strip():
                # The JUDGE failed; do not fabricate a 0 score for the candidate.
                r["na_reason"] = f"judge-failed: {completion.na_reason or 'empty judge response'}"
                continue
            if recognized_dimension_count(completion.text, r["_rubric"]) == 0:
                # Non-empty but unparseable (prose/refusal): would otherwise score
                # every dimension a fabricated 0. Mark n/a instead.
                r["na_reason"] = "judge-parse-failed"
                continue
            scores = parse_judge_response(completion.text, r["_rubric"])
            r["score"] = score_from_dimensions(scores)
            r["judge_scores"] = scores
            r["judged_by"] = judge_model
            judged_ok += 1

        agreement = self._calibrate(judged, jconf, rng)
        for r in judged:
            r.pop("_rubric", None)
        return {"judged": judged_ok, "calibration": agreement}

    def _calibrate(self, judged: list[dict], jconf: dict, rng) -> dict:
        cal_provider = jconf.get("calibration_provider")
        cal_model = jconf.get("calibration_model")
        rate = float(jconf.get("calibration_rate", 0.1))
        if not (cal_provider and cal_model) or rate <= 0:
            return {"enabled": False}
        scored = [r for r in judged if r.get("judge_scores") is not None]
        k = max(1, round(rate * len(scored))) if scored else 0
        sample = rng.sample(scored, min(k, len(scored))) if scored else []
        judge2 = build_provider(cal_provider, jconf.get("calibration_overrides"))
        diffs = []
        for r in sample:
            if not self.counter.try_reserve("judge"):
                break
            prompt = build_judge_prompt(
                r.get("task_type") or "task", r["prompt"], r.get("_rubric") or [], r["response"]
            )
            completion = judge2.run_item(cal_model, prompt)
            if completion.is_na or not completion.text.strip():
                continue  # calibration judge failed; exclude, do not fabricate
            scores2 = parse_judge_response(completion.text, r.get("_rubric") or [])
            s2 = score_from_dimensions(scores2)
            diffs.append(abs(r["score"] - s2))
        mad = sum(diffs) / len(diffs) if diffs else None
        return {
            "enabled": True,
            "calibration_model": cal_model,
            "n": len(diffs),
            "mean_abs_diff": round(mad, 4) if mad is not None else None,
        }

    # --- orchestration --------------------------------------------------------
    def run(self) -> dict:
        started = _now()
        local = self._local_providers()
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=max(1, len(local))) as pool:
            futures = [pool.submit(self._run_provider, p) for p in local]
            for fut in futures:
                # _run_provider already converts construction failures to n/a
                # rows; this guard catches anything unexpected so sibling
                # providers' completed rows are still written.
                try:
                    results.extend(fut.result())
                except Exception as exc:  # defensive
                    results.append({"na_reason": scrub(f"runner-error: {exc}"), "provider": "unknown",
                                    "model": "unknown", "dataset": "unknown", "item_id": "unknown",
                                    "grader": "unknown", "host": self.config.host, "prompt": "",
                                    "response": "", "score": None, "passed": None,
                                    "latency_s": 0.0, "tokens_in": None, "tokens_out": None,
                                    "attempts": 0, "task_type": None, "ts": _now(),
                                    "suite": self.config.suite})
        judge_info = self._judge_pass(results)
        for r in results:
            r.pop("_rubric", None)
        self.rows = results
        ended = _now()
        manifest = self._manifest(started, ended, judge_info)
        out_dir = self._write(results, manifest)
        return {"out_dir": str(out_dir), "manifest": manifest, "rows": results}

    def _manifest(self, started: str, ended: str, judge_info: dict) -> dict:
        models = sorted({r["model"] for r in self.rows})
        return {
            "run_id": f"{self.config.date}-{self.config.suite}",
            "suite": self.config.suite,
            "date": self.config.date,
            "hosts": [self.config.host],
            "git_sha": _git_sha(),
            "seed": self.config.seed,
            "providers": [p["name"] for p in self._local_providers()],
            "models": models,
            "datasets": {
                # run-specific count and sample_ids take precedence over the
                # fixture's full-file count.
                name: {
                    **ds.fixture_meta(name),
                    "count": len(self._items_by_dataset[name]),
                    "sample_ids": [it.id for it in self._items_by_dataset[name]],
                }
                for name in self.config.datasets
            },
            "caps": self.caps,
            "cap_usage": self.counter.snapshot(),
            "judge": judge_info,
            "started": started,
            "ended": ended,
        }

    def _write(self, rows: list[dict], manifest: dict) -> Path:
        out_dir = self.results_root / f"{self.config.date}-{self.config.suite}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / f"results-{self.config.host}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        # Host-scoped manifest so multiple hosts do not clobber each other.
        (out_dir / f"manifest-{self.config.host}.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return out_dir


def merge(result_files: list[str]) -> list[dict]:
    """Merge JSONL result files (from several hosts) into one list of rows."""
    rows: list[dict] = []
    for path in result_files:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def merge_manifests(manifest_files: list[str]) -> dict:
    """Union per-host manifests into one fleet-wide manifest for the report."""
    manifests = [json.loads(Path(p).read_text(encoding="utf-8")) for p in manifest_files]
    if not manifests:
        return {}
    base = dict(manifests[0])
    hosts, models, providers, git_shas = set(), set(), set(), set()
    cap_usage: dict = {}
    started = ended = None
    datasets: dict = {}
    for m in manifests:
        hosts.update(m.get("hosts", []))
        models.update(m.get("models", []))
        providers.update(m.get("providers", []))
        git_shas.add(m.get("git_sha", "unknown"))
        for k, v in (m.get("cap_usage") or {}).items():
            cap_usage[k] = cap_usage.get(k, 0) + v
        datasets.update(m.get("datasets", {}))
        s, e = m.get("started"), m.get("ended")
        started = s if started is None else min(started, s)
        ended = e if ended is None else max(ended, e)
    # If hosts ran divergent code, do not silently attribute the run to one sha.
    git_shas_sorted = sorted(git_shas)
    base.update({
        "hosts": sorted(hosts),
        "models": sorted(models),
        "providers": sorted(providers),
        "git_shas": git_shas_sorted,
        "git_sha": git_shas_sorted[0] if len(git_shas_sorted) == 1 else "multiple (see git_shas)",
        "cap_usage": cap_usage,
        "datasets": datasets,
        "started": started,
        "ended": ended,
    })
    return base
