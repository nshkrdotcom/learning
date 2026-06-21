# SELF-GROUND v0

This repo builds the initial SELF-GROUND experiment for negation-scope causal-mechanism evaluation. It generates matched negation minimal pairs, ranks SAE-like features by contrastive activation, runs feature-space necessity and sufficiency interventions, and writes transparent artifacts for inspection.

It does not train a report head, train a new SAE, provide a web UI, run broad-domain sweeps, or claim genuine model introspection.

## Requirements

- `uv`
- Python 3.11, managed by `uv`
- Optional GPU for real model runs

## Setup

```bash
uv sync
uv run pytest
uv run self-ground generate-negation --per-family 15 --out data/negation_pairs.jsonl
uv run self-ground run-negation --pairs data/negation_pairs.jsonl --sae-release <release> --sae-id <id> --out runs/negation_v0
```

Fast tests skip integration tests by default. To run model/SAE checks explicitly:

```bash
uv run pytest --run-integration
```

## Commands

```bash
uv run self-ground generate-negation --per-family 15 --out data/negation_pairs.jsonl
uv run self-ground run-negation --pairs data/negation_pairs.jsonl --model gpt2-small --layer blocks.8.hook_resid_post --sae-release <release> --sae-id <id> --top-k-features 20 --out runs/negation_v0
uv run self-ground summarize-run runs/negation_v0
```

## Outputs

Each run writes:

- `config.json`
- `pairs.jsonl`
- `feature_rankings.csv`
- `intervention_results.jsonl`
- `summary.csv`
- `README.md`

## Metric Notes

Candidate ranking uses:

```text
mean_activation(x_pos) + mean_activation(x_para) - mean_activation(x_neg) - mean_activation(x_decoy)
```

Necessity, sufficiency, specificity, collateral, and cleanliness are computed separately. Cleanliness weights are configurable in code via `MetricWeights`.

## Limitations

Fast tests use test-local doubles in `tests/` only. Production code does not ship mock adapters or expose a mock CLI mode. TransformerLens activation capture is implemented behind an integration test. SAELens loading is experimental because pretrained SAE identifiers and download behavior vary by environment. Full decoded reinjection into TransformerLens is not part of v0; intervention rows use real SAE feature-space deltas and should be treated as proxy causal evidence until the injection backend is hardened.
