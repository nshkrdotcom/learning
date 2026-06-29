# Construct Mismatch

This is a focused mechanistic-interpretability research codebase for testing construct mismatch in GPT-2 Small. It is not a general MI platform and intentionally excludes dashboards, SAEs, CLTs, circuit-tracer integrations, cross-model infrastructure, and additional constructs.

The core hypothesis is that interpretability methods often appear to disagree because they are not measuring the same validity target. The deliverable is a construct mismatch matrix showing how diff-in-means directions, linear probes, and activation patching behave when lexical cues, semantic stance, target-token behavior, and causal intervention are decoupled.

Sentiment analysis in GPT-style models is already well studied. Sentiment is included here only as a familiar baseline and pipeline sanity check. Certainty/uncertainty is the primary target construct because it is less saturated and better exposes mismatches between endorsed stance, lexical cue, and causal control.

The contribution is the controlled construct-decoupling design and the cross-method mismatch matrix. This project does not claim novelty for activation patching, linear probes, diff-in-means directions, sentiment steering, GPT-2 sentiment analysis, or the fact that probes can be predictive but non-causal.

## Setup

Use Python 3.11 or 3.12.

```bash
uv venv --python 3.12
uv pip install -e .
```

## Commands

```bash
uv run python scripts/inspect_tokenization.py
uv run python scripts/build_datasets.py
uv run python scripts/check_behavior.py --model gpt2-small

uv run python scripts/run_direction_experiment.py --model gpt2-small --construct certainty
uv run python scripts/run_direction_experiment.py --model gpt2-small --construct sentiment

uv run python scripts/run_probe_experiment.py --model gpt2-small --construct certainty
uv run python scripts/run_probe_experiment.py --model gpt2-small --construct sentiment

uv run python scripts/run_patching_experiment.py --model gpt2-small --construct certainty
uv run python scripts/run_patching_experiment.py --model gpt2-small --construct sentiment

uv run python scripts/make_figures.py
uv run python scripts/run_all.py
uv run pytest
```

For a faster patching pass while iterating:

```bash
uv run python scripts/run_patching_experiment.py --model gpt2-small --construct certainty --max-pairs-per-axis 1
```

## Outputs

- `artifacts/tokenization/gpt2_small_target_tokens.csv`
- `data/processed/*_{train,heldout,decoupling}.jsonl`
- `artifacts/behavior/behavior_summary.csv`
- `artifacts/directions/*`
- `artifacts/probes/*`
- `artifacts/patching/*`
- `artifacts/scoring/construct_mismatch_matrix.csv`
- `artifacts/scoring/object_classifications.csv`
- `reports/construct_mismatch_report.md`

If GPT-2 Small lacks usable behavior for a construct, the method scripts mark it as `behavior_absent` instead of forcing an interpretation.
