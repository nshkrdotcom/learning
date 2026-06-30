# 0901 Rebuild 06: Agent Execution Prompt and Final Acceptance

## Scope

Implement the first-build E002 Multi-QKV Shift Register architecture for the canonical A/B/C variants only:

```text
multi_qkv_static_3track_global
multi_qkv_train_rotation_3track_global
multi_qkv_position_rotation_3track_global
```

Do not implement softmix, warmup routing, LoRA deltas, learned routing, stochastic routing, coprime clocks, or typed streams.

## Required Files

Update model, training, config, scripts, canonical E002 configs, E002 docs/reports, schemas, and tests as listed in docs `00` through `05`.

## Required Tests

Run:

```bash
uv run pytest
uv run pytest --run-integration
```

The tests must cover formulas, global bank identity, one-track identity, causal masking, inactive-track gradients, standard invariance, train-step threading, eval/generation schedule semantics, diagnostics, config validation, and E002 experiment validation.

## Required QC

```bash
uv run ruff check .
uv run pytest
uv run pytest --run-integration
uv run scripts/validate_experiment.py --id E002_multitrack_qkv_shift_register
uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1.yaml
uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1.yaml
uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1.yaml
uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1.yaml
```

## Prohibitions

Do not fake artifacts. Do not run full 3000-step E002 experiments during implementation. Do not approve queue FULL rows. Do not weaken baseline manifest/config/checkpoint/eval/run verification.

## Final Self-Review Checklist

```text
[ ] Seven-file 0901 docset exists and has no extra files in its directory.
[ ] A/B/C formulas match doc 01 exactly.
[ ] One shared Q/K/V bank object is used by all blocks.
[ ] Standard and CP behavior remains covered by tests.
[ ] E002 canonical configs validate and inspect.
[ ] Full-run scripts exist but were not executed.
[ ] Destructive test script exists.
[ ] Reports state that full runs are manual and not yet claimed.
[ ] QC is green.
```

Commit and push only after QC is green.
