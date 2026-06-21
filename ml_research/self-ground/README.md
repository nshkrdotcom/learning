# SELF-GROUND

This repo implements the initial SELF-GROUND negation-scope milestone. It builds matched negation controls, captures real TransformerLens activations, ranks residual dimensions or configured SAELens features by negation contrast, and writes inspectable JSON/CSV artifacts.

It does not train a report head, train a new SAE, ship fake adapters in production, or claim genuine model introspection.

## Current Milestone

Implemented:

- deterministic Tier A/B negation minimal-pair generation
- control-purity scoring
- real TransformerLens activation capture for `EleutherAI/pythia-70m`
- real residual-dimension activation ranking without an SAE
- optional SAELens feature ranking when a release/id is provided
- real TransformerLens residual hook patching integration test
- feature-space proxy scoring clearly labeled as proxy

## Setup

```bash
uv sync
```

## Fast Tests

```bash
uv run pytest
```

## Real Model Check

```bash
uv run python scripts/check_real_model.py --device cpu
```

Equivalent CLI:

```bash
uv run self-ground check-real-model \
  --model EleutherAI/pythia-70m \
  --hook-point blocks.2.hook_resid_post \
  --device cpu \
  --out runs/check_real_model.json
```

## Real Residual-Dimension Ranking

```bash
uv run python scripts/run_real_activation_ranking.py \
  --device cpu \
  --out runs/real_activation_ranking_pythia70m
```

Equivalent CLI:

```bash
uv run self-ground run-activation-ranking \
  --model EleutherAI/pythia-70m \
  --hook-point blocks.2.hook_resid_post \
  --feature-source residual_dimensions \
  --pooling final_token \
  --per-family 15 \
  --top-k-features 50 \
  --device cpu \
  --out runs/real_activation_ranking_pythia70m
```

## Optional SAE Ranking

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

Optional SAE integration test:

```bash
SELF_GROUND_SAE_RELEASE=<release> \
SELF_GROUND_SAE_ID=<id> \
uv run pytest --run-integration tests/integration/test_sae_adapter_optional.py
```

## Optional Integration Tests

```bash
uv run pytest --run-integration
```

These tests load `EleutherAI/pythia-70m`, run real activation capture, run real residual-dimension ranking, and verify real TransformerLens hook patching changes logits.

## Data Generation

```bash
uv run self-ground generate-negation --per-family 15 --out data/negation_pairs.jsonl
```

## Artifact Layouts

Real model check:

- `runs/check_real_model.json`

Real activation ranking:

- `config.json`
- `pairs.jsonl`
- `activation_metadata.json`
- `feature_rankings.csv`
- `top_examples.jsonl`
- `README.md`

Proxy experiment path:

- `config.json`
- `pairs.jsonl`
- `feature_rankings.csv`
- `feature_space_proxy_results.jsonl`
- `summary.csv`
- `README.md`

## Metric Boundary

Residual/SAE ranking is real activation analysis: activations come from a real TransformerLens forward pass.

Feature-space proxy scoring is not behavioral causal intervention. It uses activation-level deltas to estimate proxy necessity, proxy sufficiency, proxy specificity, collateral proxy, and proxy cleanliness. Real causal intervention requires decoded reinjection into the model plus logit or behavioral measurement.

SAE decoded reinjection is not implemented in this milestone. The current blocker is documented in `docs/sae_intervention_blocker.md`.

## Known Limitations

- no report head
- no trained SAE
- no fake adapters or fake CLI modes in production
- residual-dimension ranking is real activation analysis, not mechanism discovery
- feature-space proxy scoring is not behavioral causal intervention
- SAE decoded reinjection remains a separate next milestone
