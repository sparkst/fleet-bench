# fleet-bench requirements

Canonical requirements referenced by config files (for example
configs/phase2.yaml) and code comments as REQ-FB-NN.

## REQ-FB-01: Layout

Package `fleetbench/` (providers, datasets, runner, graders, reporter) plus a
`bench.py` CLI. MIT license. README. CI runs unit tests with no network on
both PR and push.

## REQ-FB-02: Providers

One rate-limited worker per provider, providers running concurrently. Each
provider has its own requests-per-minute (rpm) and concurrency limit. Backoff
on HTTP 429/5xx, n/a after 3 failed attempts. Limits come from `caps.yaml`. An
unavailable model (not on the plan, missing key, cap hit) is reported as an
n/a row, never silently skipped.

## REQ-FB-03: Datasets, phase-gated

Datasets: IFEval, HumanEval (sandboxed unit-test grading), TREC-6, GSM8K, and
a sanity classification set. Fixed seeds and pinned item ids so runs are
reproducible.

- Phase 1: 3 items per dataset (vendored fixtures, no network, used by CI).
- Phase 2: ifeval 100, humaneval 164, trec6 200, gsm8k 100, sanity 30.

A config that requests more items than the vendored fixture provides must
fail loudly with a clear error (dataset name, requested count, available
count), not truncate or misbehave.

## REQ-FB-04: Judged SDLC suite

Six task types: coding, design, requirement writing, test writing, code
review, requirement review. Each item carries an instance-specific rubric of
4 dimensions scored 0-5. A blind judge model scores each item in randomized
processing order with hidden model ids. 10% of judged items are re-judged by
a second (calibration) model and the agreement (mean absolute difference) is
reported.

- Phase 1: 1 item per task type (6 items).
- Phase 2: 3 items per task type (18 items).

## REQ-FB-05: Hosts and manifest

Providers are sharded across hosts by config (`host` key per provider).
Results are written as JSON Lines per run, under
`results/<date>-<suite>/results-<host>.jsonl`, alongside a per-host
`manifest-<host>.json`. `merge` combines row files and `merge_manifests`
unions per-host manifests at report time.

## REQ-FB-06: Reports

Reports present a model x metric table, latency (median and p90), burn per
provider (requests, tokens where reported), n/a counts, and the exact prompts
used. Reports are committed under `reports/`. ASCII only, no em dashes (this
is a public repo).

## REQ-FB-07: Phases and acceptance

Phase 1 is a smoke run that publishes `reports/<date>-phase1.md` before Phase
2 runs. Acceptance per phase requires all of:

- CI green.
- QA Desk verdict recorded.
- Report committed and readable.
- Burn receipt recorded.
