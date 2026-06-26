# MechLedger

MechLedger is a Git-native research-integrity tool for mechanistic
interpretability repositories. It keeps tagged draft claims connected to claim
ledger entries, decisions, run records, artifacts, and visible scientific debt.

This `self-ground-v2/` repo now contains the Milestone 0 product surface, a
Milestone 1 run-auditor foundation, the Milestone 2 experiment/claim workflow
surface, Milestone 3 deterministic evidence assessment workflows, and the next
flat-file product surfaces for archival export, local audit records, question
tracking, external label metadata, local inspection, paper-safe reports, local
integrity checks, explicit redaction records, and run cleanup/archival.

## Scope

Implemented here:

- Claim-status DAG and SELF-GROUND status mapping
- Scientific-debt records and report views
- Claim ledger heading/YAML parser and canonical claim hash
- Decision log, ExperimentSpec, research log, and run ledger parsers
- Pure calibration, compatibility, and telemetry assessment logic
- Production Draft Guard for Markdown, LaTeX, and HTML claim tags
- Project scaffold, index, format, hook install, session close, status, and next
- Run wrapper, run IDs, alias cache, run directories, heartbeat, artifacts,
  run-ledger proposals, claim proposals, and scientific-debt reports
- Policy-aware ExperimentSpec prerequisite validation and `next` readiness views
- Accepted-decision-gated run reclassification with regenerated debt reports
- Declared-surface decision proposals, strict debt waivers, and stale claim
  proposal review
- Full `gate check` evidence assessment over registered metrics/artifacts:
  empirical nulls, paired statistics, matched controls, seed sensitivity,
  telemetry, compatibility, clean-candidate support, and scientific-debt reports
- Tier 2 convenience/register commands for calibration, telemetry, empirical-null
  plans/distributions, and paired-test results
- Explainer prediction locking/scoring for pre-intervention prediction records
- Deterministic RO-Crate metadata, reproducibility bundle, and paper appendix
  exports over canonical flat files and registered run/artifact metadata
- Local session/copilot audit records with accepted-decision review metadata
- Open-question tracking surfaced by `next`
- External label registry import/link commands; labels remain metadata, not
  evidence by default
- Local dashboard JSON and query commands over canonical flat files
- Deterministic Draft Guard suggestion and claim-language reports
- Typed optional platform record schemas/validators for PRD-defined activation,
  weight-analysis, circuit-graph, and cross-model-comparison metadata, plus
  stricter extension records for correspondence, training-dynamics, and
  remote-job metadata
- Local sync/conflict reporting with `sync status` and `sync diff`; these report
  drift and never merge state
- Explicit run/artifact redaction records, redacted artifact placeholders, and
  redacted-supporting-evidence debt
- Integrity/tamper records with accepted-decision resolution workflow
- Run pinning, dry-run garbage collection, archive-before-delete cleanup, and
  per-run reproducibility bundles
- Copilot output review for ignored local assistant artifacts, with explicit
  accept/reject/modified outcomes and committed provenance sidecars
- Conservative environment redaction and explicit SDK `external_call` event
  metadata requirements
- Minimal dependency-light SDK helpers, including a pure-Python sign-test helper
- Backfilled E001-E004 claim ledger, run ledger, research log, and reuse decision
- Milestone -1 report with dogfood results and divergences

Intentionally not implemented here:

- Heavy model execution or intervention abstractions
- Hosted dashboard/server, LLM generation/review queues, remote sync merge, broad
  redaction workflows beyond explicit local records, RDF/JSON-LD as canonical
  storage, or model-execution platform surfaces
- Heavy ML dependencies such as `torch`, `transformer_lens`, `sae_lens`,
  `numpy`, or `scipy`

## Quick Start

```bash
uv run mechledger --help
uv run mechledger init
uv run mechledger draft check research/paper/draft.md
uv run mechledger draft check --staged
uv run mechledger index --check
uv run mechledger index --check --staged
uv run mechledger run -- python scripts/your_experiment.py
uv run mechledger gate check latest
uv run mechledger calibration check latest
uv run mechledger telemetry check latest
uv run mechledger prediction lock research/predictions/sae_12300.json
uv run mechledger prediction score PRED001 --against-run latest
uv run mechledger export appendix --out research/paper/mechledger_appendix.md
uv run mechledger dashboard data --out .mechledger/dashboard/data.json
uv run mechledger questions list
uv run mechledger copilot list
uv run mechledger sync status
uv run mechledger integrity check
uv run mechledger pin latest
uv run mechledger gc --keep-last 100 --keep-pinned
uv run mechledger experiment validate research/experiments/*.md
uv run mechledger next
uv run mechledger status
```

See [docs/USAGE.md](docs/USAGE.md) for Draft Guard setup, wrapping scripts, SDK
usage, artifacts, aliases, Tier 2 evidence registration, prediction locking,
exports, sessions, copilot provenance review, questions, labels, query commands,
language reports, optional records, sync reporting, redaction, integrity records,
run lifecycle,
crystallization, claim review, staged-mode contracts, read-only cache fallback,
run reclassification safety, decisions, debt waivers, and no-ML end-to-end
assessment examples. The PRD coverage map lives in
[docs/PRD_COVERAGE_0430_0432.md](docs/PRD_COVERAGE_0430_0432.md) with the
machine-readable backing file
[docs/prd_coverage_0430_0432.json](docs/prd_coverage_0430_0432.json). The
prompt-completion ledger lives in
[docs/PRD_COMPLETION_LEDGER_0430_0432.md](docs/PRD_COMPLETION_LEDGER_0430_0432.md)
and [docs/prd_completion_ledger_0430_0432.json](docs/prd_completion_ledger_0430_0432.json);
it records implemented, partial, deferred, ambiguous, and out-of-scope
dispositions per required surface.

## Layout

```text
src/mechledger/core/          claim status, debt, and claim ledger logic
src/mechledger/assessments/   pure assessment helpers over JSON-shaped inputs
src/mechledger/draftguard.py  production Draft Guard
src/mechledger/cli.py         Typer command surface
src/mechledger/sdk/           dependency-light active-run helpers
research/logs/                dogfood claim ledger, decision log, source copies
tests/                        TDD coverage and real/synthetic fixtures
```

## QC

Run from this directory:

```bash
uv run pytest
uv run ruff check .
uv run mechledger --help
uv run mechledger index --check
```

## Boundaries

MechLedger does not execute interventions, intercept arbitrary network calls,
discover artifacts outside registered paths or run-local artifact directories,
merge SQLite, perform remote sync merge, enforce untagged paper claims, detect
constants hidden in Python source, make scientific truth decisions, import
heavy ML libraries in core, compute activations/circuits/weights/platform
records, compute p-values/null statistics, verify citations/recompute reported
statistics, or use RO-Crate/RDF/SQLite as canonical storage. Users register
metrics and artifacts from their own research environment. Optional platform
records are metadata validation/export records only. MechLedger may allow work
to continue with unresolved scientific debt, but it surfaces that debt. External
labels are metadata by default, and session/copilot records require human review
before becoming canonical interpretation. Copilot review records provenance
only; it does not call an LLM, verify scientific truth, promote claims, invent
run results, or waive debt. Failed or cancelled runs cannot be promoted into
evidence-supporting run classes; rerun them or record them as negative evidence.
