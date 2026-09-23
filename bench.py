#!/usr/bin/env python3
"""fleet-bench command line interface.

Commands:
  run     Run a benchmark from a run-config YAML and write JSONL + manifest.
  report  Render a markdown report from a run directory.
  info    List available providers and datasets.

Keys come from environment variables only (GROQ_API_KEY, HETZNER_EXPERIMENTS_TOKEN;
the Claude/Codex/Copilot legs use their logged-in CLIs). Nothing secret is read
from or written to this repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from fleetbench import datasets as ds
from fleetbench.providers import provider_names
from fleetbench.reporter import render
from fleetbench.runner import RunConfig, Runner, merge, merge_manifests


def cmd_run(args) -> int:
    config = RunConfig.from_dict(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    runner = Runner(config, caps_path=args.caps, results_root=args.results)
    result = runner.run()
    print(f"wrote {len(result['rows'])} rows to {result['out_dir']}")
    print(f"manifest: {result['out_dir']}/manifest-{config.host}.json")
    return 0


def cmd_report(args) -> int:
    run_dir = Path(args.run_dir)
    manifest_files = [str(p) for p in sorted(run_dir.glob("manifest-*.json"))]
    if not manifest_files:
        print(f"no manifest-*.json in {run_dir}", file=sys.stderr)
        return 2
    manifest = merge_manifests(manifest_files)
    jsonl_files = [str(p) for p in sorted(run_dir.glob("results-*.jsonl"))]
    if not jsonl_files:
        print(f"no results-*.jsonl in {run_dir}", file=sys.stderr)
        return 2
    rows = merge(jsonl_files)
    # Persist the merged manifest for reference alongside the per-host ones.
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    markdown = render(rows, manifest)
    out_path = Path(args.out) if args.out else Path("reports") / f"{manifest['run_id']}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    print(f"wrote report to {out_path}")
    return 0


def cmd_info(args) -> int:
    print("providers:", ", ".join(provider_names()))
    print("datasets:", ", ".join(ds.DATASET_NAMES))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="fleetbench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a benchmark from a run-config YAML")
    p_run.add_argument("--config", required=True, help="path to run-config YAML")
    p_run.add_argument("--caps", default=None, help="path to caps.yaml (default: repo caps.yaml)")
    p_run.add_argument("--results", default="results", help="results root directory")
    p_run.set_defaults(func=cmd_run)

    p_report = sub.add_parser("report", help="render a markdown report from a run directory")
    p_report.add_argument("--run-dir", required=True, help="results/<date>-<suite> directory")
    p_report.add_argument("--out", default=None, help="output markdown path")
    p_report.set_defaults(func=cmd_report)

    p_info = sub.add_parser("info", help="list providers and datasets")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
