# 0901 Rebuild 05: Configs, Scripts, and Manual Run Loop

## Canonical Configs

```text
configs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1.yaml
configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1.yaml
configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1.yaml
configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1.yaml
```

## Manual Full-Run Scripts

```text
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_standard_refactor_control.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_static_global.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_train_rotation_global.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_position_rotation_global.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_all_full_initial.sh
scripts/experiments/E002_multitrack_qkv_shift_register/compare_initial_full_runs.sh
```

Each full-run script verifies data, refuses existing run artifacts by default, trains, verifies, runs eval loss, generation, bounded HellaSwag, summarizes, and verifies final artifacts.

## Manual Run Loop

From the repo root:

```bash
pwd
git status --short
uv run scripts/verify_cuda.py
uv run scripts/verify_data.py --config configs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1.yaml
uv run scripts/validate_experiment.py --id E002_multitrack_qkv_shift_register

scripts/experiments/E002_multitrack_qkv_shift_register/run_full_standard_refactor_control.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_static_global.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_train_rotation_global.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_position_rotation_global.sh

scripts/experiments/E002_multitrack_qkv_shift_register/compare_initial_full_runs.sh
```

Interpretation loop:

1. Human launches one full run at a time.
2. Human waits for completion outside implementation.
3. Human verifies artifacts.
4. Human runs destructive test.
5. Human runs comparison.
6. Human updates results docs.
7. Human decides continue, rerun, or kill.

No fake run summaries, comparison JSON, eval artifacts, or result claims are allowed.
