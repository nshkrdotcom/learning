# MechLedger

MechLedger is a Git-native research-integrity tool for mechanistic
interpretability repositories. It keeps tagged draft claims connected to claim
ledger entries, decisions, run records, artifacts, and visible scientific debt.

This `self-ground-v2/` repo now contains the Milestone 0 product surface, a
Milestone 1 run-auditor foundation, the Milestone 2 experiment/claim workflow
surface, and Milestone 3 deterministic evidence assessment workflows built on
the Milestone -1 SELF-GROUND extraction kernel.

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
- Minimal dependency-light SDK helpers
- Backfilled E001-E004 claim ledger, run ledger, research log, and reuse decision
- Milestone -1 report with dogfood results and divergences

Intentionally not implemented here:

- Heavy model execution or intervention abstractions
- Hosted dashboard, copilot review queues, RO-Crate export, sync merge, garbage
  collection, redaction, prediction locking, or Tier 3+ platform surfaces
- Heavy ML dependencies such as `torch`, `transformer_lens`, `sae_lens`,
  `numpy`, or `scipy`

## Quick Start

```bash
uv run mechledger --help
uv run mechledger init
uv run mechledger draft check research/paper/draft.md
uv run mechledger index --check
uv run mechledger run -- python scripts/your_experiment.py
uv run mechledger gate check latest
uv run mechledger experiment validate research/experiments/*.md
uv run mechledger next
uv run mechledger status
```

See [docs/USAGE.md](docs/USAGE.md) for Draft Guard setup, wrapping scripts, SDK
usage, artifacts, aliases, crystallization, claim review, decisions, debt
waivers, and assessment examples.

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

MechLedger does not execute interventions, discover artifacts outside registered
paths or run-local artifact directories, merge SQLite, enforce untagged paper
claims, detect constants hidden in Python source, make scientific truth
decisions, import heavy ML libraries in core, compute p-values/null statistics,
or verify citations/recompute reported statistics. Users register metrics and
artifacts from their own research environment. MechLedger may allow work to
continue with unresolved scientific debt, but it surfaces that debt.
