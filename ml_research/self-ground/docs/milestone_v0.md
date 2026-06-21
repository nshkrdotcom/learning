# Milestone v0: Real Negation Activation Ranking

## Completed Capabilities

- Deterministic negation minimal-pair generation for copula, do-support, existential, and modal templates.
- Stable SHA-256-based pair IDs.
- Control-purity scoring for matched negation controls.
- Real TransformerLens activation capture through `TransformerLensModelAdapter`.
- Real residual-dimension feature extraction with explicit `final_token` and `mean` pooling.
- Real residual-dimension ranking for negation contrast.
- Optional SAELens encoding path when a concrete SAE release/id is configured.
- Feature-space proxy metrics clearly labeled as proxy.
- Real TransformerLens hook patching API and integration test.

## Commands Run For This Milestone

```bash
uv sync
uv run pytest
uv run python scripts/check_real_model.py --device cpu
uv run python scripts/run_real_activation_ranking.py --device cpu --per-family 1 --top-k-features 5 --out runs/test_real_activation_ranking
```

When environment resources permit:

```bash
uv run pytest --run-integration
```

## Artifacts Produced

Real model check:

- `runs/check_real_model.json`

Real activation ranking:

- `runs/test_real_activation_ranking/config.json`
- `runs/test_real_activation_ranking/pairs.jsonl`
- `runs/test_real_activation_ranking/activation_metadata.json`
- `runs/test_real_activation_ranking/feature_rankings.csv`
- `runs/test_real_activation_ranking/top_examples.jsonl`
- `runs/test_real_activation_ranking/README.md`

## What Counts As Real Evidence

The residual-dimension ranking path uses real model activations from `EleutherAI/pythia-70m` captured at `blocks.2.hook_resid_post`. The ranking table is real activation contrast evidence over matched negation examples.

The residual patching integration test uses real TransformerLens hooks and verifies that zeroing the residual stream at the hook point changes model logits.

## What Remains Proxy

`feature_space_proxy_results.jsonl` contains activation-space proxy deltas. These are not behavioral causal interventions because decoded activation reinjection and logit/behavior measurement are not implemented for SAE features in this milestone.

## Next Milestone

The next milestone is decoded SAE feature intervention:

1. Select a tested SAE release/id compatible with `EleutherAI/pythia-70m` and `blocks.2.hook_resid_post`.
2. Confirm encode/decode tensor shapes against real activations.
3. Modify selected SAE features.
4. Decode back to residual space.
5. Patch decoded residual activations into TransformerLens.
6. Measure logit-contrast changes on negation controls.
