# MechLedger

MechLedger is a dependency-light, Git-native research integrity system for mechanistic interpretability repositories. It keeps tagged draft claims connected to claim ledger entries, run records, artifacts, decisions, and visible scientific debt.

This `self-ground-v2` directory is a clean extraction from the original SELF-GROUND work. It intentionally does not copy the TransformerLens/SAELens execution harness. The reusable part is the audit kernel: claim status discipline, causal-language guardrails, evidence debt, run records, and artifact provenance.

## Quick Start

```bash
uv sync --dev
uv run mechledger init
uv run mechledger draft check research/paper/draft.md
uv run mechledger run -- python scripts/your_experiment.py
uv run mechledger gate check latest
```

## Core Commands

- `mechledger init`: create `research/` and `.mechledger/project.json`.
- `mechledger draft check`: lint `[CLAIM:C001]`, `\claim{C001}`, and `<!-- CLAIM:C001 -->` tags against `research/logs/claim_ledger.md`.
- `mechledger index --check`: validate claim, decision, experiment, run ledger, and local run files.
- `mechledger run -- <command>`: wrap an arbitrary native command and capture run metadata.
- `mechledger attach RUN_ID PATH`: register an artifact explicitly.
- `mechledger artifact annotate RUN_ID A001 --claim-relevance supporting`: mark reviewed evidence.
- `mechledger gate check RUN_ID`: emit a scientific debt report.
- `mechledger next`: classify experiments as ready or blocked from machine-readable prerequisites.

## Non-Goals

MechLedger does not execute interventions for you, discover arbitrary artifacts, merge SQLite, enforce untagged paper claims, detect constants hidden in source code, make scientific truth decisions, or verify citations/statistics. It records declared evidence relationships and keeps scientific debt visible.
