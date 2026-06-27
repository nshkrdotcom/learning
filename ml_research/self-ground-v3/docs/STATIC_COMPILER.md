# Static Compiler

The static compiler runs cheap structural checks before claim-bearing causal verification.

It is not causal evidence. It answers a narrower question: is the hypothesis algebraically plausible enough to justify spending compute on exact verification?

## Command

```bash
uv run mwb compile hypothesis docs/fixtures/hypothesis_phase5.json
```

The command writes:

```text
.mechanism/static_compiler/latest_static_compile.json
```

and indexes the report in SQLite tables `static_compiler_reports` and `static_check_results`.

## Input Contract

A hypothesis may include a `static_compiler` object:

```json
{
  "static_compiler": {
    "tensor_space_ref": "space_resid_post",
    "unembedding_space_ref": "space_resid_post",
    "target_token_ids": [10],
    "foil_token_ids": [20],
    "decoder_vector": [1.0, 0.0, 0.0],
    "unembedding": {
      "10": [1.0, 0.0, 0.0],
      "20": [0.0, 1.0, 0.0]
    },
    "dictionary": {
      "feature_id": "unit_fixture",
      "decoder_vectors": {
        "unit_fixture": [1.0, 0.0, 0.0],
        "unit_clean_neighbor": [0.0, 0.0, 1.0]
      }
    },
    "activation_density": {
      "target": 0.1,
      "control": 0.1,
      "max_ratio": 1.5
    }
  }
}
```

The compiler uses provided vectors directly. Older metadata fields such as `decoder_unembed_projection_score` are not treated as compiler evidence when `static_compiler` is present.

## Checks

`decoder_unembed_projection` computes:

```text
direction = normalize_l2(decoder_vector)
contrast = normalize_l2(mean(target unembedding vectors) - mean(foil unembedding vectors))
score = direction dot contrast
```

Default thresholds:

- `pass`: score >= 0.03
- `warn`: 0.0 < score < 0.03
- `fail`: score <= 0.0

`neighbor_interference` scans decoder-vector cosine against dictionary neighbors.

Default thresholds:

- `fail`: nearest-neighbor cosine >= 0.8
- `warn`: nearest-neighbor cosine >= 0.6
- `pass`: below warning threshold

`activation_density` compares target/control density symmetrically. If either side is more than `max_ratio` times the other, the check warns with `activation_density_mismatch`.

## Gate

The compiler uses weakest-link aggregation:

- `PASS`: all checks pass.
- `WEAK`: at least one warning and no failures.
- `FAIL`: at least one failure.

Claim-bearing verification blocks on `FAIL`. Diagnostic-only verification can still run, but cannot be cited as strong causal evidence.

## Repair

`mwb repair-index` restores compiler reports and individual check rows from `.mechanism/static_compiler/*.json`.

## Claim Boundary

A passing compiler report only establishes structural plausibility. It cannot support causal language, promote a claim, or substitute for prediction-locked intervention evidence.
