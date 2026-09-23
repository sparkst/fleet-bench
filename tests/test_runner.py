"""Runner end to end with fake providers (no network, no CLI).

Covers: deterministic grading inline, a model reported n/a when not on the plan,
host sharding, the blind judge pass, calibration, judge failure NOT fabricated as
a 0 score, the cost-multiplier skip, provider construction failure reported (not
aborting the run), manifest count consistency, host-scoped manifest, and merge.
"""

from __future__ import annotations

import json
import re

import pytest

from fleetbench import runner as runner_mod
from fleetbench.providers.base import Completion
from fleetbench.runner import RunConfig, Runner, merge, merge_manifests


class FakeProvider:
    def __init__(self, name, available, judge_fails=False, judge_prose=False):
        self.name = name
        self._available = available
        self.judge_fails = judge_fails
        self.judge_prose = judge_prose
        self.concurrency = 1

    def available_models(self):
        return list(self._available)

    def run_item(self, model, prompt):
        if "CANDIDATE_RESPONSE_BEGIN" in prompt:  # a judge prompt
            if self.judge_fails:
                return Completion(text="", latency_s=0.0, na_reason="error after retries: boom")
            if self.judge_prose:  # non-empty but unparseable
                return Completion(text="I cannot score this response.", latency_s=0.02)
            dims = re.findall(r"^- ([^:]+):", prompt, re.M)
            return Completion(
                text=json.dumps({d: 4 for d in dims}), latency_s=0.02,
                tokens_in=3, tokens_out=4,
            )
        return Completion(text="fruit", latency_s=0.01, tokens_in=5, tokens_out=6)


def _patch(monkeypatch, judge_fails=False, judge_prose=False):
    def fake_build(name, overrides=None):
        if name == "anthropic":
            return FakeProvider("anthropic", ["j"], judge_fails=judge_fails, judge_prose=judge_prose)
        return FakeProvider("groq", ["m-ok"])
    monkeypatch.setattr(runner_mod, "build_provider", fake_build)


def _config():
    return RunConfig.from_dict({
        "suite": "t", "date": "2026-01-01", "host": "primary", "seed": 0,
        "datasets": {"sanity": 2, "sdlc": 1},
        "providers": [
            {"name": "groq", "host": "primary", "models": ["m-ok", "m-missing"]},
            {"name": "copilot", "host": "other"},  # sharded away; not run here
        ],
        "judge": {
            "provider": "anthropic", "model": "j",
            "calibration_provider": "groq", "calibration_model": "cal",
            "calibration_rate": 0.5,
        },
    })


def test_runner_end_to_end(monkeypatch, tmp_path):
    _patch(monkeypatch)
    r = Runner(_config(), results_root=str(tmp_path))
    result = r.run()
    rows = result["rows"]

    assert {row["provider"] for row in rows} == {"groq"}  # host sharding
    assert len(rows) == 6  # 2 models x (2 sanity + 1 sdlc)

    missing = [row for row in rows if row["model"] == "m-missing"]
    assert len(missing) == 3
    assert all(row["na_reason"] == "model-unavailable-on-plan" for row in missing)

    sanity_ok = [r for r in rows if r["model"] == "m-ok" and r["dataset"] == "sanity"]
    assert all(r["score"] is not None for r in sanity_ok)

    sdlc = [r for r in rows if r["model"] == "m-ok" and r["dataset"] == "sdlc"][0]
    assert sdlc["score"] == pytest.approx(0.8)
    assert sdlc["judged_by"] == "j"
    assert "_rubric" not in sdlc

    manifest = result["manifest"]
    assert manifest["run_id"] == "2026-01-01-t"
    # count is the run count, consistent with sample_ids (not the fixture size).
    for name, meta in manifest["datasets"].items():
        assert meta["count"] == len(meta["sample_ids"])
    assert manifest["judge"]["judged"] == 1
    assert manifest["judge"]["calibration"]["enabled"] is True
    assert manifest["hosts"] == ["primary"]

    out_dir = tmp_path / "2026-01-01-t"
    assert (out_dir / "manifest-primary.json").exists()
    jsonl = out_dir / "results-primary.jsonl"
    assert jsonl.exists()
    assert len(merge([str(jsonl)])) == 6


def test_judge_failure_not_fabricated(monkeypatch, tmp_path):
    _patch(monkeypatch, judge_fails=True)
    r = Runner(_config(), results_root=str(tmp_path))
    result = r.run()
    sdlc = [row for row in result["rows"] if row["dataset"] == "sdlc" and row["model"] == "m-ok"][0]
    # The judge failed: the row is n/a, NOT a fabricated 0 score.
    assert sdlc["score"] is None
    assert sdlc["na_reason"].startswith("judge-failed")
    assert "judge_scores" not in sdlc
    assert result["manifest"]["judge"]["judged"] == 0


def test_judge_parse_failure_not_fabricated(monkeypatch, tmp_path):
    _patch(monkeypatch, judge_prose=True)
    r = Runner(_config(), results_root=str(tmp_path))
    result = r.run()
    sdlc = [row for row in result["rows"] if row["dataset"] == "sdlc" and row["model"] == "m-ok"][0]
    # Non-empty but unparseable judge output -> n/a, NOT a fabricated 0 score.
    assert sdlc["score"] is None
    assert sdlc["na_reason"] == "judge-parse-failed"
    assert result["manifest"]["judge"]["judged"] == 0


def test_judge_not_configured_marks_rows_na(monkeypatch, tmp_path):
    def fake_build(name, overrides=None):
        return FakeProvider("groq", ["m-ok"])
    monkeypatch.setattr(runner_mod, "build_provider", fake_build)
    cfg = RunConfig.from_dict({
        "suite": "t", "date": "2026-01-01", "host": "primary", "seed": 0,
        "datasets": {"sdlc": 1},
        "providers": [{"name": "groq", "host": "primary", "models": ["m-ok"]}],
        # no judge configured
    })
    rows = Runner(cfg, results_root=str(tmp_path)).run()["rows"]
    assert all(row["na_reason"] == "judge-not-configured" for row in rows)
    assert all(row["score"] is None for row in rows)


def test_cost_multiplier_skip(monkeypatch, tmp_path):
    def fake_build(name, overrides=None):
        return FakeProvider("copilot", ["m-ok"])
    monkeypatch.setattr(runner_mod, "build_provider", fake_build)
    cfg = RunConfig.from_dict({
        "suite": "t", "date": "2026-01-01", "host": "primary", "seed": 0,
        "datasets": {"sanity": 1},
        "providers": [{"name": "copilot", "host": "primary", "models": ["m-ok"]}],
    })
    r = Runner(cfg, results_root=str(tmp_path))
    # copilot caps carry skip_multiplier_gt: 5; force the item above it.
    for item in r._items_by_dataset["sanity"]:
        item.cost_multiplier = 99.0
    rows = r.run()["rows"]
    assert all(row["na_reason"] == "cost-multiplier-exceeded" for row in rows)


def test_provider_construction_failure_is_reported(monkeypatch, tmp_path):
    def fake_build(name, overrides=None):
        raise KeyError("unknown provider")
    monkeypatch.setattr(runner_mod, "build_provider", fake_build)
    cfg = RunConfig.from_dict({
        "suite": "t", "date": "2026-01-01", "host": "primary", "seed": 0,
        "datasets": {"sanity": 1},
        "providers": [{"name": "groq", "host": "primary", "models": ["x"]}],
    })
    rows = Runner(cfg, results_root=str(tmp_path)).run()["rows"]
    assert rows and all(row["na_reason"].startswith("provider-error") for row in rows)


def test_internal_concurrency_runs_all_items_and_respects_cap(monkeypatch, tmp_path):
    import time

    from fleetbench.providers.base import Completion as C
    from fleetbench.providers.base import Provider

    class ConcurrentProvider(Provider):
        name = "groq"

        def __init__(self, **kw):
            super().__init__(**kw)
            self.concurrency = 4

        def available_models(self):
            return ["m"]

        def call(self, model, prompt):
            time.sleep(0.01)  # force real thread interleaving
            return C(text="fruit", latency_s=0.01)

    monkeypatch.setattr(runner_mod, "build_provider", lambda name, overrides=None: ConcurrentProvider())
    cfg = RunConfig.from_dict({
        "suite": "t", "date": "2026-01-01", "host": "primary", "seed": 0,
        "datasets": {"sanity": 3},
        "providers": [{"name": "groq", "host": "primary", "models": ["m"]}],
    })
    r = Runner(cfg, results_root=str(tmp_path))
    # Cap below the item count: the shared CapCounter must not be exceeded under
    # real concurrent access.
    r.caps = {"providers": {"groq": {"max_calls": 2}}}
    r.counter = runner_mod.CapCounter(r.caps)
    rows = r.run()["rows"]
    assert len(rows) == 3  # every item produced a row
    served = [row for row in rows if not row["na_reason"]]
    capped = [row for row in rows if row["na_reason"] == "cap-exceeded"]
    assert len(served) == 2 and len(capped) == 1  # cap held exactly under threads


def test_merge_manifests_flags_git_sha_divergence(tmp_path):
    a = tmp_path / "manifest-h1.json"
    b = tmp_path / "manifest-h2.json"
    a.write_text(json.dumps({"run_id": "r", "hosts": ["h1"], "git_sha": "AAAA"}), encoding="utf-8")
    b.write_text(json.dumps({"run_id": "r", "hosts": ["h2"], "git_sha": "BBBB"}), encoding="utf-8")
    m = merge_manifests([str(a), str(b)])
    assert m["git_shas"] == ["AAAA", "BBBB"]
    assert m["git_sha"] == "multiple (see git_shas)"


def test_merge_manifests_unions(tmp_path):
    a = tmp_path / "manifest-h1.json"
    b = tmp_path / "manifest-h2.json"
    a.write_text(json.dumps({
        "run_id": "r", "hosts": ["h1"], "models": ["m1"], "providers": ["groq"],
        "cap_usage": {"groq": 3}, "datasets": {"sanity": {"count": 1}},
        "started": "2026-01-01T00:00:00Z", "ended": "2026-01-01T00:05:00Z",
    }), encoding="utf-8")
    b.write_text(json.dumps({
        "run_id": "r", "hosts": ["h2"], "models": ["m2"], "providers": ["codex"],
        "cap_usage": {"groq": 2, "codex": 1}, "datasets": {"gsm8k": {"count": 1}},
        "started": "2026-01-01T00:01:00Z", "ended": "2026-01-01T00:09:00Z",
    }), encoding="utf-8")
    m = merge_manifests([str(a), str(b)])
    assert m["hosts"] == ["h1", "h2"]
    assert m["models"] == ["m1", "m2"]
    assert m["cap_usage"] == {"groq": 5, "codex": 1}
    assert set(m["datasets"]) == {"sanity", "gsm8k"}
    assert m["started"] == "2026-01-01T00:00:00Z"
    assert m["ended"] == "2026-01-01T00:09:00Z"


def test_merge_combines_multiple_files(tmp_path):
    f1 = tmp_path / "a.jsonl"
    f2 = tmp_path / "b.jsonl"
    f1.write_text(json.dumps({"x": 1}) + "\n", encoding="utf-8")
    f2.write_text(json.dumps({"x": 2}) + "\n" + json.dumps({"x": 3}) + "\n", encoding="utf-8")
    assert [m["x"] for m in merge([str(f1), str(f2)])] == [1, 2, 3]
