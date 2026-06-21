# Milestone v0 / Phase 1

## Phase Goal

Phase 1 asks:

Can we build a real, reproducible pipeline that finds negation-contrastive
internal residual dimensions in a small real model, then smoke-patches those
dimensions and measures whether the model's logits change in a
negation-relevant way?

## Completed Capabilities

- Deterministic negation minimal-pair generation for four template families.
- Stable SHA-256-based pair IDs.
- Control-purity scoring for matched negation controls.
- Real TransformerLens activation capture through `TransformerLensModelAdapter`.
- Real residual-dimension extraction with explicit `final_token` and `mean` pooling.
- Real residual-dimension ranking for negation contrast.
- Real residual-dimension smoke patching through TransformerLens hooks.
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

Real residual smoke diagnostic:

- `runs/real_residual_intervention_pythia70m/config.json`
- `runs/real_residual_intervention_pythia70m/selected_features.json`
- `runs/real_residual_intervention_pythia70m/intervention_results.jsonl`
- `runs/real_residual_intervention_pythia70m/summary.csv`
- `runs/real_residual_intervention_pythia70m/README.md`

## Interpretation Guide

Real activation ranking means the table is computed from actual residual stream activations captured from a real TransformerLens model.

Real residual smoke diagnostic means selected residual dimensions are patched
through real TransformerLens hooks, the model is rerun, and logit-contrast
changes are measured. It verifies the patching pipeline, but it is
diagnostic-only and cannot feed `candidate_evidence` or
`strong_candidate_evidence`.

The key Phase 1 metric in `intervention_results.jsonl` is:

```text
specificity_score = mean(abs(delta[x_pos]), abs(delta[x_para]))
                  - mean(abs(delta[x_neg]), abs(delta[x_decoy]))
```

This checks whether the residual smoke patch moves negation conditions more than
matched affirmation/decoy controls.

## Limitations

- Residual dimensions are basis-dependent.
- Residual ranking is not sparse-feature mechanism discovery.
- Residual smoke patching is not SAE decoded intervention or paper-facing
  evidence.
- Feature-space proxy scoring is not behavioral causal evidence.
- No report head is trained.
- No broad generalization claim is made.

## Current Phase Map

Current repo numbering:

- Phase 1: real residual activation ranking and residual smoke diagnostics.
- Phase 2: decoded SAE feature intervention. See `docs/phase2_sae_intervention.md`.
- Phase 3: multi-task token-contrast evaluation and candidate evidence reports.
  See `docs/phase3_token_contrast_evaluation.md`.
