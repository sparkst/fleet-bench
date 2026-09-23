"""End-to-end CLI test: bench.py run + report against fake providers (offline)."""

from __future__ import annotations

import json

import pytest

import bench
from fleetbench import runner as runner_mod
from fleetbench.providers.base import Completion


class FakeProvider:
    def __init__(self, name, available):
        self.name = name
        self._available = available
        self.concurrency = 1

    def available_models(self):
        return list(self._available)

    def run_item(self, model, prompt):
        return Completion(text="fruit", latency_s=0.01, tokens_in=5, tokens_out=6)


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(runner_mod, "build_provider", lambda name, overrides=None: FakeProvider("groq", ["m1"]))


def _write_config(path):
    path.write_text(
        "suite: t\n"
        "date: '2026-01-01'\n"
        "host: primary\n"
        "seed: 0\n"
        "datasets:\n  sanity: 2\n"
        "providers:\n  - name: groq\n    host: primary\n    models: ['m1']\n"
        "judge: {}\n",
        encoding="utf-8",
    )


def test_info_command_lists_providers_and_datasets(capsys):
    assert bench.main(["info"]) == 0
    out = capsys.readouterr().out
    assert "groq" in out and "sanity" in out


def test_run_then_report(patched, tmp_path):
    cfg = tmp_path / "run.yaml"
    _write_config(cfg)
    results = tmp_path / "results"

    rc = bench.main(["run", "--config", str(cfg), "--results", str(results)])
    assert rc == 0
    run_dir = results / "2026-01-01-t"
    assert (run_dir / "manifest-primary.json").exists()
    rows = [json.loads(l) for l in (run_dir / "results-primary.jsonl").read_text().splitlines()]
    assert len(rows) == 2

    out_md = tmp_path / "report.md"
    rc = bench.main(["report", "--run-dir", str(run_dir), "--out", str(out_md)])
    assert rc == 0
    text = out_md.read_text(encoding="utf-8")
    assert "fleet-bench report: 2026-01-01-t" in text
    assert "## Accuracy by dataset" in text
    # The merged manifest is persisted for reference.
    assert (run_dir / "manifest.json").exists()


def test_report_missing_dir_returns_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert bench.main(["report", "--run-dir", str(empty)]) == 2
