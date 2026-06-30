# E002 Multitrack QKV Shift Register Plan

## Hypothesis

Multiple learned Q/K/V tracks are cycled, shifted, or cross-wired across layers and/or training steps to test whether scheduled QKV track interoperation changes learning behavior beyond static multi-track capacity.

## Non-Claim Boundaries

This experiment may only support a local finding under Attention Lab's small-GPT FineWeb-Edu setup. It must not claim broad transformer superiority, reasoning improvement, or scaling behavior.

## Planned Variants

- `standard_30m_seed1`: runnable standard-attention control.
- `multi_qkv_static_3track_30m_seed1`: planned static multi-track capacity control.
- `multi_qkv_train_and_layer_shift_3track_30m_seed1`: planned train-step and layer shift schedule.
- `multi_qkv_train_shift_3track_30m_seed1`: planned train-step shift ablation.
- `multi_qkv_layer_shift_3track_30m_seed1`: planned layer shift ablation.
- `multi_qkv_softmix_3track_30m_seed1`: planned soft mixing control.
- `multi_qkv_train_shift_warmup_3track_30m_seed1`: planned warmup schedule ablation.

## Fixed Contract

All direct comparisons must use the same FineWeb-Edu 100M manifest, GPT-2 tokenizer, train/val shards, model size, optimizer, LR schedule, seed, batch construction, eval cadence, checkpoint cadence, and verification commands unless a config is explicitly labeled diagnostic.

## Mechanism Diagnostics

Future QKV variants must emit diagnostics that can be evaluated by `queue.mechanism_check: qkv_track_activity`, including at least one of:

- `track_gradient_norm`
- `per_track_gradient_norm`
- `branch_off_logit_delta`
- `track_output_delta`

## Run Matrix

The first implementation pass should run unit and tiny integration tests only. Full 3000-step runs must be launched manually from approved queue entries or scripts after configs, diagnostics, and control dependencies validate.

## Success Criteria

- Candidate configs instantiate only after architecture code exists.
- Standard control remains unchanged.
- Mechanism diagnostics show nonzero track activity.
- Full-run comparisons pass manifest-aware verify/eval/summarize checks.

## Kill Criteria

- Track diagnostics remain zero or missing.
- Shifted variants are slower and no better than static multi-track control.
- Lambda/wiring controls diverge when they should be equivalent.
- Any full run fails manifest, checkpoint, or verifier checks.
