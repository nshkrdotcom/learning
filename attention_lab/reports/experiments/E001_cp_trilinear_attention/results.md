# E001 CP Trilinear Attention Results

Status: implementation and tiny integration complete; full 3000-step comparison runs
prepared but not executed in this pass.

## Claim Boundary

No scientific performance claim is made yet. Once manual full runs are executed, the
only permitted claim boundary is:

```text
Under this repository's local ~30M GPT FineWeb-Edu 100M training setup, the
CP-trilinear score branch either did or did not show promising evidence relative to
standard attention and CP-bilinear controls.
```

## Run Matrix

| Variant | Config | Run Directory | Status |
| --- | --- | --- | --- |
| Standard | `standard_30m_seed1.yaml` | `runs/experiments/E001_cp_trilinear_attention/standard_30m_seed1` | not_run |
| Standard refactor control | `standard_refactor_control_30m_seed1.yaml` | `runs/experiments/E001_cp_trilinear_attention/standard_refactor_control_30m_seed1` | not_run |
| CP bilinear r8 | `cp_bilinear_r8_30m_seed1.yaml` | `runs/experiments/E001_cp_trilinear_attention/cp_bilinear_r8_30m_seed1` | not_run |
| CP trilinear r8 | `cp_trilinear_r8_30m_seed1.yaml` | `runs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_30m_seed1` | not_run |
| CP trilinear r8 lambda0 | `cp_trilinear_r8_lambda0_30m_seed1.yaml` | `runs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_lambda0_30m_seed1` | not_run |

## Commands Actually Run In This Pass

```bash
uv sync
uv run pytest
uv run ruff check .
uv run scripts/validate_experiment.py --id E001_cp_trilinear_attention
uv run scripts/verify_data.py --data_root data/fineweb_edu_100m --manifest data/fineweb_edu_100m/manifest.json --verify_hashes
uv run scripts/inspect_model_config.py --config configs/experiments/E001_cp_trilinear_attention/standard_30m_seed1.yaml
uv run scripts/inspect_model_config.py --config configs/experiments/E001_cp_trilinear_attention/cp_bilinear_r8_30m_seed1.yaml --baseline-config configs/experiments/E001_cp_trilinear_attention/standard_30m_seed1.yaml
uv run scripts/inspect_model_config.py --config configs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_30m_seed1.yaml --baseline-config configs/experiments/E001_cp_trilinear_attention/standard_30m_seed1.yaml
uv run scripts/inspect_model_config.py --config configs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_lambda0_30m_seed1.yaml --baseline-config configs/experiments/E001_cp_trilinear_attention/standard_30m_seed1.yaml
```

Tiny CP integration runs were executed through `pytest` temporary directories, not as
scientific evidence.

## Implementation Result

- CP-bilinear attention module implemented.
- CP-trilinear attention module implemented.
- Config validation accepts canonical `cp_bilinear` and `cp_trilinear` configs and
  rejects invalid CP lambda settings.
- Attention diagnostics are emitted for CP variants at configured cadence.
- Candidate model parameter deltas are inspectable.
- Manual full-run scripts are executable and prepared.

## QC Results

```text
pytest: 92 passed in 7.92s
ruff: All checks passed!
validate_experiment: ok=True, config_count=5, runnable_config_count=5, unimplemented_config_count=0
verify_data: manifest verified for data/fineweb_edu_100m/manifest.json
historical baseline verify_run: ok=True, data_manifest=True
bash -n scripts/experiments/E001_cp_trilinear_attention/*.sh: passed
```

## Full-Run Result

Not executed in this pass. Run:

```bash
scripts/experiments/E001_cp_trilinear_attention/run_all_full.sh
scripts/experiments/E001_cp_trilinear_attention/compare_full_runs.sh
```

Then update this report and `run_index.*` with the verified results.

## Scientific Interpretation

No scientific interpretation is available until the full direct-comparison runs finish,
their checkpoints reload, bounded HellaSwag/generation paths run, summaries exist, and
`verify_run.py` passes with `--expect-data-manifest`.

## Kill Criteria Status

No full-run kill criteria have been triggered because the full runs were not executed.
Implementation-level checks did not show NaNs, broken verification, missing diagnostics,
or failed checkpoint reload in tiny integration tests.
