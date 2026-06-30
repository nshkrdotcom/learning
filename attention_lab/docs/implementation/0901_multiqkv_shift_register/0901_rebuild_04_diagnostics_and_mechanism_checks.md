# 0901 Rebuild 04: Diagnostics and Mechanism Checks

## Diagnostic Fields

Multi-QKV attention diagnostics must emit:

```text
active_track_index
active_track_counts
per_track_gradient_norm
per_track_qkv_weight_norm
per_track_output_norm
track_entropy
route_formula
uses_global_bank
layer_idx
step
position_routing_enabled
eval_freeze_mode
```

`active_track_index` is an integer for layer-wide routes and null for per-position routes. `active_track_counts` counts tokens routed to each track. `per_track_gradient_norm` and `per_track_qkv_weight_norm` are per track. `track_entropy` is computed from observed route counts.

## Mechanism Check

The `qkv_track_activity` mechanism check must require nonzero activity, such as nonzero `per_track_gradient_norm`, `track_output_delta`, or equivalent future QKV diagnostic. Field presence alone is not enough.

Missing or degenerate diagnostics block non-standard FULL promotion unless explicitly overridden by queue policy.

## Destructive Test

The destructive/off-route check lives at:

```text
scripts/qkv_track_destructive_test.py
```

It loads a checkpoint, runs a natural forward pass, then forces selected tracks off or swapped and writes logit/loss deltas to:

```text
runs/experiments/E002_multitrack_qkv_shift_register/<run_name>/evals/qkv_track_destructive_test.json
```

The script must not invent metrics. It writes output only from an actual checkpoint/data forward pass.
