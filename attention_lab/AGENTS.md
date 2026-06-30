# AGENTS.md

This repo is Attention Lab: a small GPT pretraining harness for controlled local
attention experiments.

## Start Here

1. Read `README.md` for the operator flow.
2. Read `docs/architecture_experiment_contract.md` before changing architecture code.
3. Read `docs/architecture_variant_checklist.md` before adding a new attention module.
4. Read `docs/guides/experiment_queue_discipline_checklist.md` before queueing or
   preparing long runs.
5. Read the relevant experiment plan under `docs/experiments/`.

## Baseline Rules

- Do not rewrite the trainer for architecture variants.
- Do not weaken manifest checks, config validation, checkpointing, eval, or
  `verify_run.py`.
- Keep standard attention intact unless an experiment explicitly includes a
  standard-refactor control.
- Select architectures through config and the attention registry.

## Experiment Rules

- New attention modules go under `src/attention_lab/models/attention/`.
- Experiment configs go under `configs/experiments/<EXPERIMENT_ID>/`.
- Reports go under `reports/experiments/<EXPERIMENT_ID>/`.
- Full-run evidence must come from actual train/eval/summarize/verify commands.
- Do not claim a full run completed unless `verify_run.py` passed with the required
  flags.

## Queue Rules

- `docs/guides/experiment_queue_discipline_checklist.md` contains both implementation
  acceptance checks and reusable run-operation checklists.
- The top `Implementation Status` section records what has been built.
- The detailed unchecked boxes are templates/gates for future queue work and for each
  experiment run; they are not expected to all be globally checked forever.
- Long full runs should be launched from a frozen source state/worktree. The user may
  handle freeze/run execution outside this working copy.

## Required QC

Before committing source/docs changes, normally run:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run scripts/validate_experiment.py --id E001_cp_trilinear_attention
uv run scripts/verify_data.py --data_root data/fineweb_edu_100m --manifest data/fineweb_edu_100m/manifest.json --verify_hashes
```
