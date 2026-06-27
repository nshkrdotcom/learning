# Local MI Lab

This is a local mechanistic-interpretability practice lab for learning by running small, clean, reproducible experiments.

It is not SELF-GROUND. It does not continue negation-scope. It is not a platform, workbench, protocol engine, research OS, or generic provenance system.

The first goal is induction, logit-lens, and activation-patching practice on small local models. GPT-2 small is the default because mature TransformerLens tooling and fast iteration matter more than scale at this stage. Pythia-410M is included as a secondary model after the GPT-2 small workflow works.

Scripts are the reproducible path. Notebooks are inspection front-ends that load configs and artifacts produced by scripts.

## Quick Start

```bash
uv sync
uv run ruff check .
uv run pytest
uv run python scripts/check_model_capability.py --config configs/gpt2_small_induction.yaml
uv run python scripts/build_toy_prompts.py --config configs/gpt2_small_induction.yaml
uv run python scripts/run_baseline_behavior.py --config configs/gpt2_small_induction.yaml
uv run python scripts/cache_activations.py --config configs/gpt2_small_induction.yaml --run runs/<run_id>
uv run python scripts/run_logit_lens.py --config configs/gpt2_small_induction.yaml --run runs/<run_id>
uv run python scripts/run_attention_patterns.py --config configs/gpt2_small_induction.yaml --run runs/<run_id>
uv run python scripts/summarize_run.py --run runs/<run_id>
```

## Controls Workflow

Use this after the basic induction run works. The goal is to catch false positives: raw attention to a previous token is not enough, and a candidate head should separate positive induction prompts from controls.

```bash
uv run python scripts/build_toy_prompts.py --config configs/gpt2_small_induction_controls.yaml
uv run python scripts/run_baseline_behavior.py --config configs/gpt2_small_induction_controls.yaml
uv run python scripts/cache_activations.py --config configs/gpt2_small_induction_controls.yaml --run runs/<run_id>
uv run python scripts/run_logit_lens.py --config configs/gpt2_small_induction_controls.yaml --run runs/<run_id>
uv run python scripts/run_attention_patterns.py --config configs/gpt2_small_induction_controls.yaml --run runs/<run_id>
uv run python scripts/summarize_run.py --run runs/<run_id>
```

Failure is useful. If controls also score highly, the current prompt family or metric is not specific enough.

Activation patching is separate and should be run only after baseline behavior, activation capture, logit lens, and attention-pattern inspection are working:

```bash
uv run python scripts/run_activation_patching.py --config configs/gpt2_small_clean_corrupt_tiny.yaml
```

## Repository Shape

- `configs/`: small YAML configs for known-good exercises.
- `scripts/`: reproducible entry points that produce CSV, JSON, PNG, and Markdown artifacts.
- `src/local_mi_lab/`: reusable package code.
- `notebooks/`: lightweight artifact inspection front-ends.
- `docs/`: learning path, experiment ideas, resource plan, and experiment log.
- `data/`, `runs/`, `reports/`: generated local artifacts, ignored by git except placeholders.

## Scope

The first pass does not require SAEs, Gemma, nnsight, Neuronpedia, dashboards, databases, agent-managed IPython, or large model runs.

This is still practice work. Attention inspection is descriptive. Residual-stream cache and logit-lens artifacts are descriptive. Patching is causal only for the selected prompt pair, component, position, and metric. No broad mechanism claim is allowed.

Creating this repo is not MI progress. Passing tests is not MI progress. Capability checks are not MI progress. The first actual learning result is the baseline behavior table and the human inspection of activations, attention patterns, and patching that follows.
