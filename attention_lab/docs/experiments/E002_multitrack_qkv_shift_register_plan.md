# E002 Multi-QKV Shift Register Plan

## Hypothesis

Deterministic hard-switched routing over a globally shared three-track bundled Q/K/V bank may change local small-GPT learning behavior beyond standard attention and beyond static global-bank capacity alone.

This experiment does not claim broad transformer superiority, reasoning improvement, or scaling behavior.

## First-Build Scope

Implemented canonical variants:

```text
standard_refactor_control_30m_seed1
multi_qkv_static_3track_global_30m_seed1
multi_qkv_train_rotation_3track_global_30m_seed1
multi_qkv_position_rotation_3track_global_30m_seed1
```

Old skeleton variants remain `status: experimental_unimplemented` and are not first-build evidence.

## Architecture Contract

All Multi-QKV first-build variants use one globally shared bank:

```text
Q_bank[0..2], K_bank[0..2], V_bank[0..2]
```

Q/K/V are bundled:

```text
Q = Q_bank[active_track]
K = K_bank[active_track]
V = V_bank[active_track]
```

The active track is deterministic, hard-switched, not learned, and not content-dependent.

## A/B/C Formulas

```text
track_count = 3
```

A, static global:

```text
active_track(layer_idx, step, pos) = layer_idx mod 3
```

B, train rotation:

```text
training: active_track(layer_idx, step, pos) = (layer_idx + step) mod 3
eval:     active_track(layer_idx, step, pos) = layer_idx mod 3
```

C, position rotation:

```text
active_track(layer_idx, step, pos) = (layer_idx + pos) mod 3
```

## Fixed Contract

Direct comparisons use the same FineWeb-Edu 100M manifest, GPT-2 tokenizer, train/val shards, optimizer, LR schedule, seed, batch construction, eval cadence, checkpoint cadence, and verifier/eval/summarize commands.

Because the Q/K/V bank is global rather than per-layer, Multi-QKV variants have fewer parameters than standard attention. This is reported explicitly. The primary controls for B and C are A/static-global plus the standard refactor control.

## Diagnostics

Multi-QKV variants emit diagnostics to:

```text
runs/experiments/E002_multitrack_qkv_shift_register/<run_name>/evals/attention_diagnostics.jsonl
```

Required fields include active track index/counts, per-track gradient/weight/output norms, track entropy, route formula, global-bank flag, layer index, step, position-routing flag, and eval-freeze flag.

The queue mechanism check is:

```yaml
queue:
  mechanism_check: qkv_track_activity
```

## Manual Full-Run Boundary

Implementation QC does not include full 3000-step E002 runs. Full runs are manual/operator work from a frozen source state using:

```text
scripts/experiments/E002_multitrack_qkv_shift_register/
```

No result report may claim completion until train/eval/generation/HellaSwag/summarize/final verify commands actually pass.

## Success Criteria

- Candidate configs instantiate and validate.
- Standard control remains unchanged.
- Mechanism diagnostics show nonzero selected-track activity.
- Full-run comparisons pass manifest-aware verify/eval/summarize checks.
- B/C are interpreted primarily relative to A/static-global.

## Kill Criteria

- Track diagnostics are zero, missing, or degenerate.
- B or C is slower and no better than A/static-global.
- Eval/generation for B uses training-step rotation.
- Any full run fails manifest, checkpoint, or verifier checks.
