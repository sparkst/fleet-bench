# fleet-bench

A parallel, rate-limit-aware benchmark of coding harnesses and models on public,
verifiable datasets. It runs many providers at once, each throttled to its own
rate limit, grades the results with objective graders (plus a blind LLM judge for
open-ended tasks), and writes a plain-markdown report you can read on GitHub.

The results are meant to be shared: reports live under `reports/` and the exact
prompts, dataset ids, and grading logic are in the repo so anyone can reproduce
a run.

## What it measures

- Accuracy on verifiable datasets: IFEval (rule following), HumanEval (code that
  passes unit tests), TREC-6 (question classification), GSM8K (grade-school math),
  and a trivial authored sanity set.
- A judged SDLC suite across six task types (coding, design, requirement writing,
  test writing, code review, requirement review), scored by a blind LLM judge on
  four 0-to-5 rubric dimensions per item, with a calibration re-judge by a second
  judge model.
- Latency (median and p90), and burn per provider (requests, and tokens where the
  provider reports them). CLI-backed providers include CLI startup in latency; the
  report footnotes this.

## Providers

| provider | how it is called | keys / login |
|---|---|---|
| Groq | HTTP (OpenAI-compatible) | `GROQ_API_KEY` |
| Hetzner | HTTP (OpenAI-compatible), thinking off by default | `HETZNER_EXPERIMENTS_TOKEN` |
| Anthropic | `claude -p --model <model>` (prompt on stdin) | logged-in `claude` CLI |
| Codex | `codex exec --skip-git-repo-check --json --model <model>` (prompt on stdin) | logged-in `codex` CLI |
| Copilot | `copilot -s --allow-all-tools --model <model> -p <prompt>` | logged-in `copilot` CLI |
| OpenRouter | HTTP (OpenAI-compatible) | `OPENROUTER_API_KEY` |
| Gemini | HTTP (OpenAI-compatible) | `GEMINI_API_KEY` |
| Cloudflare Workers AI | HTTP (OpenAI-compatible) | `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID` |

Keys are read from environment variables only; nothing secret is stored in this
repo. A model that is not available on a plan is reported as `n/a`, never silently
skipped. The Cloudflare provider also needs `CLOUDFLARE_ACCOUNT_ID` to build its
base URL (the account id is never hardcoded); it raises a clear error at
construction time if that variable is unset. Runs are invoked under
`doppler run -p claude-code -c dev -- fleetbench ...` so the keys are injected
from the secrets manager rather than an interactive shell.

Safety note: the Copilot leg runs with `--allow-all-tools` (required for its
non-interactive mode), which grants the agent tool and shell execution on the host
that runs the leg. The HumanEval grader likewise executes model-generated code.
Both are only process-isolated today (scrubbed environment, resource limits, a
throwaway working directory), which is adequate for the authored Phase 1 smoke
prompts. Before Phase 2 loads full external datasets, run the agentic CLI legs and
the HumanEval grader inside a network-disabled, filesystem-restricted container.

## Install

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```
# List providers and datasets.
fleetbench info

# Run a benchmark from a run config.
fleetbench run --config configs/phase1.yaml

# Render the markdown report from the run directory.
fleetbench report --run-dir results/2026-09-17-phase1
```

The run writes `results/<date>-<suite>/results-<host>.jsonl` and a
`manifest.json` (models, datasets, pinned sample ids, caps, git sha, timestamps).
`report` merges the per-host JSONL files and writes `reports/<run-id>.md`.

### Rate limits, caps, and budgets

Each provider runs one rate-limited worker; providers run concurrently, and up to
`concurrency` items run in parallel within a provider (from `caps.yaml`; the default
is 1). Per-model requests-per-minute and per-provider call caps come from `caps.yaml`.
The defaults are call counts, which are safe to publish; the operator may layer
private budget limits (for example a share of a 5-hour or weekly quota) at run time.
On a 429 or 5xx the worker backs off and retries, and gives up as `n/a` after three
failed attempts on an item.

Cap semantics: `max_calls` is enforced per `bench.py run` invocation and counts
items attempted, not raw HTTP requests (a single item may make up to three attempts
under retry). It is a fast in-process guard, not a cross-run daily ledger. The
Hetzner provider additionally writes a per-call ledger line so the fleet can count a
daily budget across runs; that ledger is accounting for an external counter, not
enforced by this tool.

### Sharding CLI legs across hosts

The CLI providers can run on a host where their CLI is logged in, reached over ssh,
while the API legs run locally. Set the ssh host with an environment variable so no
machine name is committed:

```
export FLEETBENCH_CLI_SSH_HOST=<your-ssh-host>
```

The Codex and Copilot providers then run their command on that host. `report`
merges the JSONL from every host into one report.

## Phases

- Phase 1 (smoke): 3 items per dataset and 1 per SDLC task type, across every
  available model. It must run end to end and publish `reports/<date>-phase1.md`
  before Phase 2. Phase 1 is non-discriminative by design (see the caveat banner
  reporter.py prepends to every phase1 report); do not read its near-universal
  1.000 scores as a model ranking.
- Phase 2 (full): IFEval 100, HumanEval 164, TREC-6 200, GSM8K 100, sanity 30, and
  3 items per SDLC task type. Phase 2 loaders fetch the full public datasets with
  pinned ids; the graders are identical to Phase 1.

The Phase 1 smoke ships vendored fixtures under `data/phase1/` so unit tests and CI
run with no network. Each fixture states its provenance and license.

## Datasets and licenses

| dataset | source | license |
|---|---|---|
| IFEval | Google IFEval instruction-following rules | Apache-2.0 |
| HumanEval | OpenAI HumanEval | MIT |
| TREC-6 | TREC question classification | research use |
| GSM8K | OpenAI GSM8K | MIT |
| sanity | authored in this repo | MIT |
| SDLC suite | authored in this repo | MIT |

Phase 1 smoke items for IFEval, HumanEval, TREC-6, and GSM8K are authored in this
repo in each dataset's format and taxonomy, labeled as such in the fixture; the full
public datasets with their pinned ids are used in Phase 2.

## Add a provider

1. Add a module under `fleetbench/providers/`. For an OpenAI-compatible HTTP API,
   subclass `OpenAICompatProvider` (see `groq.py`, `openrouter.py`, `gemini.py`,
   and `cloudflare.py`). For a CLI, subclass `CLIProvider` and set `base_command`
   with a `{model}` token (see `codex_cli.py`).
2. Read any key from an environment variable in the constructor.
3. Register the class in `fleetbench/providers/__init__.py`.
4. Add a caps entry in `caps.yaml`.

## Add a dataset

1. Add a fixture `data/phase1/<name>.json` with `provenance`, `license`, and items.
2. Add a builder in `fleetbench/datasets/__init__.py` that turns the fixture into
   `Item` records with the prompt the model sees and the grader key.
3. If it needs a new grader, add it under `fleetbench/graders/` and register it in
   `fleetbench/graders/__init__.py`.

## Tests and CI

```
pytest
```

All unit tests run offline (no network, no CLI). CI runs them on every push and PR.
A secret-pattern test guards against committing secrets, checks for internal
identifiers, and enforces ASCII-only, em-dash-free text. Running `gitleaks` locally
as a pre-commit hook is a recommended extra.

## License

MIT. See `LICENSE`.
