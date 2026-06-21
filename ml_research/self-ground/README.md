# SELF-GROUND

SELF-GROUND Phase 1 is a real negation-scope activation and residual-intervention pipeline for a small TransformerLens model. It generates matched negation controls, captures real activations from `EleutherAI/pythia-70m`, ranks raw residual dimensions by negation contrast, patches selected residual dimensions, and measures real logit-contrast deltas.

This repo does not train a report head, train a new SAE, ship fake production adapters, or claim mechanism discovery.

## Phase 1 Command Sequence

```bash
uv sync
uv run pytest
uv run python scripts/check_real_model.py --device cpu
uv run python scripts/run_real_activation_ranking.py \
  --device cpu \
  --out runs/real_activation_ranking_pythia70m
uv run python scripts/run_real_residual_intervention.py \
  --ranking-dir runs/real_activation_ranking_pythia70m \
  --device cpu \
  --out runs/real_residual_intervention_pythia70m
```

Equivalent CLI commands:

```bash
uv run self-ground check-real-model \
  --model EleutherAI/pythia-70m \
  --hook-point blocks.2.hook_resid_post \
  --device cpu \
  --out runs/check_real_model.json

uv run self-ground run-activation-ranking \
  --model EleutherAI/pythia-70m \
  --hook-point blocks.2.hook_resid_post \
  --feature-source residual_dimensions \
  --pooling final_token \
  --per-family 15 \
  --top-k-features 50 \
  --device cpu \
  --out runs/real_activation_ranking_pythia70m

uv run self-ground run-residual-intervention \
  --ranking-dir runs/real_activation_ranking_pythia70m \
  --model EleutherAI/pythia-70m \
  --hook-point blocks.2.hook_resid_post \
  --top-k-features 5 \
  --operation zero \
  --device cpu \
  --out runs/real_residual_intervention_pythia70m
```

## What Is Real In Phase 1

- real TransformerLens activation capture
- real residual-dimension negation ranking
- real residual-dimension intervention through TransformerLens hooks
- real logit-contrast deltas after patching residual activations
- inspectable JSONL/CSV artifacts

## What Is Not Claimed

- no SAE decoded intervention
- no sparse-feature causal intervention
- no mechanism discovery
- no report-head training
- no broad generalization claim
- no genuine model introspection claim

## Optional Integration Tests

```bash
uv run pytest --run-integration
```

These tests load `EleutherAI/pythia-70m`, capture real activations, run real residual-dimension ranking, and verify real residual patching changes logits. The optional SAELens test skips unless `SELF_GROUND_SAE_RELEASE` and `SELF_GROUND_SAE_ID` are set.

## Data Generation

```bash
uv run self-ground generate-negation --per-family 15 --out data/negation_pairs.jsonl
```

## Optional SAE Ranking

SAE ranking is available only when a real SAELens release/id is configured:

```bash
uv run self-ground run-activation-ranking \
  --model EleutherAI/pythia-70m \
  --hook-point blocks.2.hook_resid_post \
  --feature-source sae \
  --sae-release <release> \
  --sae-id <id> \
  --pooling final_token \
  --per-family 15 \
  --top-k-features 50 \
  --device cpu \
  --out runs/real_sae_ranking_pythia70m
```

SAE decoded reinjection is not implemented in Phase 1. See `docs/sae_intervention_blocker.md`.

## Artifacts

Real model check:

- `runs/check_real_model.json`

Real activation ranking:

- `config.json`
- `pairs.jsonl`
- `activation_metadata.json`
- `feature_rankings.csv`
- `top_examples.jsonl`
- `README.md`

Real residual intervention:

- `config.json`
- `selected_features.json`
- `intervention_results.jsonl`
- `summary.csv`
- `README.md`

Legacy/internal proxy scoring, where used in tests, writes `feature_space_proxy_results.jsonl` and is explicitly not behavioral causal evidence.

## Interpretation Boundary

Residual-dimension ranking is real activation analysis over a fixed residual basis. It can identify dimensions whose activation contrasts with negation controls, but it is not mechanism discovery.

Residual intervention is real behavioral measurement at the logit level: the model is rerun with selected residual dimensions patched and logit contrasts are compared. Raw residual dimensions are basis-dependent and should not be interpreted as sparse semantic features.

Phase 2 is decoded SAE feature intervention: encode activations, modify selected SAE features, decode to residual space, patch the model, and measure logit/behavior changes.
