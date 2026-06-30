# 0901 Rebuild 03: TDD and QC Plan

## Required Test Files

```text
tests/test_attention_multi_qkv_global.py
tests/test_model_multi_qkv_global.py
tests/test_train_multi_qkv_step_position_threading.py
tests/test_validate_experiment_e002.py
tests/test_qkv_rotation_diagnostics.py
```

## Required Coverage

1. Shape tests.
2. Causal masking tests.
3. One-track identity test against standard attention.
4. Global-bank sharing test proving blocks reference the same bank object.
5. A formula test: `layer_idx mod 3`.
6. B formula test: `(layer_idx + step) mod 3` during train, `layer_idx mod 3` during eval.
7. C formula test: `(layer_idx + pos) mod 3` with per-position routing.
8. Standard attention invariance under step/position threading.
9. Diagnostics emission test.
10. Train-loop integration test proving step reaches attention modules.
11. Eval/generation integration test proving eval mode uses correct schedule.
12. Config validation test for new fields.

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

No full 3000-step run is automated in this implementation pass.
