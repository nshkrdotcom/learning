# SELF-GROUND

SELF-GROUND is a negation-scope interpretability experiment harness. The repo now has a Phase 1 real residual-dimension pipeline and Phase 2 decoded SAE intervention infrastructure. It does not claim complete SELF-GROUND, broad mechanism discovery, or genuine model introspection.

## Phase 1 Recap

Phase 1 implements:

- deterministic negation minimal pairs,
- real TransformerLens activation capture,
- real residual-dimension activation ranking,
- real residual-dimension intervention through TransformerLens hooks,
- real logit-contrast deltas after residual patching.

Phase 1 does not claim sparse SAE mechanisms. Raw residual dimensions are basis-dependent.

## Phase 2 Goal

Phase 2 adds decoded SAE feature intervention:

```text
real model activations
  -> real SAELens encode
  -> modify selected SAE feature activations
  -> decode back to residual space
  -> patch the real TransformerLens model
  -> rerun logits
  -> write intervention artifacts
```

If no compatible SAE release/id is available, Phase 2 writes a precise compatibility artifact and does not write fabricated intervention rows.

## Setup And Fast Tests

```bash
uv sync
uv run pytest
```

## Phase 1 Commands

```bash
uv run python scripts/check_real_model.py --device cpu

uv run python scripts/run_real_activation_ranking.py \
  --device cpu \
  --out runs/real_activation_ranking_pythia70m

uv run python scripts/run_real_residual_intervention.py \
  --ranking-dir runs/real_activation_ranking_pythia70m \
  --device cpu \
  --out runs/real_residual_intervention_pythia70m
```

## Phase 2 SAE Compatibility

The repo has one verified small SAE path:

- model: `EleutherAI/pythia-70m-deduped`
- hook point: `blocks.2.hook_resid_post`
- SAE release: `pythia-70m-deduped-res-sm`
- SAE id: `blocks.2.hook_resid_post`

The Phase 2 evidence run is documented in `docs/phase2_run_evidence.md`.

```bash
uv run python scripts/check_sae_compatibility.py \
  --model EleutherAI/pythia-70m-deduped \
  --hook-point blocks.2.hook_resid_post \
  --sae-release pythia-70m-deduped-res-sm \
  --sae-id blocks.2.hook_resid_post \
  --device cpu \
  --out runs/check_sae_compatibility.json
```

Equivalent CLI:

```bash
uv run self-ground check-sae-compatibility \
  --model EleutherAI/pythia-70m-deduped \
  --hook-point blocks.2.hook_resid_post \
  --sae-release pythia-70m-deduped-res-sm \
  --sae-id blocks.2.hook_resid_post \
  --device cpu \
  --out runs/check_sae_compatibility.json
```

## Phase 2 SAE Ranking

```bash
uv run python scripts/run_real_activation_ranking.py \
  --model EleutherAI/pythia-70m-deduped \
  --hook-point blocks.2.hook_resid_post \
  --device cpu \
  --feature-source sae \
  --sae-release pythia-70m-deduped-res-sm \
  --sae-id blocks.2.hook_resid_post \
  --out runs/real_sae_ranking_pythia70m
```

## Phase 2 SAE Intervention

```bash
uv run python scripts/run_real_sae_intervention.py \
  --ranking-dir runs/real_sae_ranking_pythia70m \
  --out runs/real_sae_intervention_pythia70m \
  --model EleutherAI/pythia-70m-deduped \
  --hook-point blocks.2.hook_resid_post \
  --sae-release pythia-70m-deduped-res-sm \
  --sae-id blocks.2.hook_resid_post \
  --top-k-features 5 \
  --operation ablate \
  --patch-mode delta \
  --device cpu
```

Equivalent CLI:

```bash
uv run self-ground run-sae-intervention \
  --ranking-dir runs/real_sae_ranking_pythia70m \
  --out runs/real_sae_intervention_pythia70m \
  --model EleutherAI/pythia-70m-deduped \
  --hook-point blocks.2.hook_resid_post \
  --sae-release pythia-70m-deduped-res-sm \
  --sae-id blocks.2.hook_resid_post \
  --top-k-features 5 \
  --operation ablate \
  --patch-mode delta \
  --device cpu
```

## Optional Integration Tests

Real SAE integration tests require:

```bash
export SELF_GROUND_SAE_RELEASE=pythia-70m-deduped-res-sm
export SELF_GROUND_SAE_ID=blocks.2.hook_resid_post
uv run pytest --run-integration
```

Without those variables, SAE integration tests skip and the Phase 2 blocker workflow remains the executable path.

## Artifacts

Phase 1 model check:

- `runs/check_real_model.json`

Phase 1 residual ranking:

- `config.json`
- `pairs.jsonl`
- `activation_metadata.json`
- `feature_rankings.csv`
- `top_examples.jsonl`
- `README.md`

Phase 1 residual intervention:

- `config.json`
- `selected_features.json`
- `intervention_results.jsonl`
- `summary.csv`
- `README.md`

Phase 2 compatibility:

- `runs/check_sae_compatibility.json`

Phase 2 SAE intervention:

- `config.json`
- `compatibility.json`
- `selected_features.json`
- `intervention_results.jsonl`
- `summary.csv`
- `README.md`

If compatibility fails, Phase 2 writes only `config.json`, `compatibility.json`, and `README.md`.

## Interpretation Boundaries

Feature-space proxy arithmetic is not causal evidence.

Residual-dimension intervention is real TransformerLens intervention evidence, but it is not SAE feature intervention and not mechanism discovery.

Decoded SAE intervention is real sparse-feature intervention only when compatibility succeeds and the run writes decoded SAE intervention artifacts.
