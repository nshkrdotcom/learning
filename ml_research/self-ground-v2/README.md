# MechLedger Milestone -1

This directory is the Milestone -1 extraction kernel for MechLedger, built in
the `self-ground-v2/` repo path. It extracts SELF-GROUND's research-integrity
logic into dependency-light, framework-free Python modules that can be tested
against real SELF-GROUND artifacts before a CLI surface exists.

## Scope

Implemented here:

- Claim-status DAG and SELF-GROUND status mapping
- Scientific-debt records and report views
- Claim ledger heading/YAML parser and canonical claim hash
- Pure calibration, compatibility, and telemetry assessment logic
- Draft Guard prototype for tagged Markdown sentence windows
- Backfilled E001-E004 claim ledger and reuse decision log
- Milestone -1 report with dogfood results and divergences

Intentionally not implemented here:

- CLI entrypoint or `typer.Typer()` app
- Run wrapper, heartbeat, SDK, run directory writer, alias cache, file locking,
  SQLite indexer, hooks, dashboard, or copilot surface
- Heavy ML dependencies such as `torch`, `transformer_lens`, `sae_lens`,
  `numpy`, or `scipy`

## Layout

```text
src/mechledger/core/          claim status, debt, and claim ledger logic
src/mechledger/assessments/   pure assessment helpers over JSON-shaped inputs
src/mechledger/draftguard_proto.py
research/logs/                dogfood claim ledger, decision log, source copies
tests/                        TDD coverage and real/synthetic fixtures
```

## QC

Run from this directory:

```bash
uv run pytest
uv run ruff check .
```

Milestone 0 should wrap this kernel only after the Milestone -1 report is
reviewed.
