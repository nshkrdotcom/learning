# Milestone v0 / Phase 1

## Phase Goal

Phase 1 asks:

Can we build a real, reproducible pipeline that finds negation-contrastive internal residual dimensions in a small real model, then intervenes on those dimensions and measures whether the model's logits change in a negation-relevant way?

## Completed Capabilities

- Deterministic negation minimal-pair generation for four template families.
- Stable SHA-256-based pair IDs.
- Control-purity scoring for matched negation controls.
- Real TransformerLens activation capture through `TransformerLensModelAdapter`.
- Real residual-dimension extraction with explicit `final_token` and `mean` pooling.
- Real residual-dimension ranking for negation contrast.
- Real residual-dimension intervention through TransformerLens hooks.
- Real logit-contrast measurement before and after residual patching.
- Optional SAELens encoding for ranking when a concrete release/id is configured.
- Feature-space proxy metrics kept explicitly separate from real intervention artifacts.

## Commands

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

Smaller verification run:

```bash
uv run python scripts/run_real_activation_ranking.py \
  --device cpu \
  --per-family 1 \
  --top-k-features 5 \
  --out runs/test_real_activation_ranking
uv run python scripts/run_real_residual_intervention.py \
  --ranking-dir runs/test_real_activation_ranking \
  --device cpu \
  --top-k-features 2 \
  --out runs/test_real_residual_intervention
```

## Artifacts

Real model check:

- `runs/check_real_model.json`

Real activation ranking:

- `runs/real_activation_ranking_pythia70m/config.json`
- `runs/real_activation_ranking_pythia70m/pairs.jsonl`
- `runs/real_activation_ranking_pythia70m/activation_metadata.json`
- `runs/real_activation_ranking_pythia70m/feature_rankings.csv`
- `runs/real_activation_ranking_pythia70m/top_examples.jsonl`
- `runs/real_activation_ranking_pythia70m/README.md`

Real residual intervention:

- `runs/real_residual_intervention_pythia70m/config.json`
- `runs/real_residual_intervention_pythia70m/selected_features.json`
- `runs/real_residual_intervention_pythia70m/intervention_results.jsonl`
- `runs/real_residual_intervention_pythia70m/summary.csv`
- `runs/real_residual_intervention_pythia70m/README.md`

## Interpretation Guide

Real activation ranking means the table is computed from actual residual stream activations captured from a real TransformerLens model.

Real residual intervention means selected residual dimensions are patched through real TransformerLens hooks, the model is rerun, and logit-contrast changes are measured.

The key Phase 1 metric in `intervention_results.jsonl` is:

```text
specificity_score = mean(abs(delta[x_pos]), abs(delta[x_para]))
                  - mean(abs(delta[x_neg]), abs(delta[x_decoy]))
```

This tests whether the residual patch moves negation conditions more than matched affirmation/decoy controls.

## Limitations

- Residual dimensions are basis-dependent.
- Residual ranking is not sparse-feature mechanism discovery.
- Residual intervention is not SAE decoded intervention.
- Feature-space proxy scoring is not behavioral causal evidence.
- No report head is trained.
- No broad generalization claim is made.

## Next Phase

Phase 2 is decoded SAE feature intervention:

1. Select a tested SAE release/id compatible with the model and hook point.
2. Confirm encode/decode tensor shapes against real activations.
3. Modify selected SAE features.
4. Decode back to residual space.
5. Patch decoded residual activations into TransformerLens.
6. Measure logit-contrast changes on negation controls.
