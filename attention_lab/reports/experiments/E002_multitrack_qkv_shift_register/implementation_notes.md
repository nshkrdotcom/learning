# E002 Implementation Notes

## Scope

Implemented first-build global Multi-QKV variants:

```text
multi_qkv_static_3track_global
multi_qkv_train_rotation_3track_global
multi_qkv_position_rotation_3track_global
```

Not implemented in this pass:

```text
softmix
warmup routing
LoRA deltas
learned routing
stochastic routing
coprime decoupled Q/K/V clocks
typed content/control/binding streams
```

## Parameter Counts

From `scripts/inspect_model_config.py`:

| run | attention type | excluding positional | including positional | trainable | attention projection | non-attention | delta vs standard |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard_refactor_control_30m_seed1 | standard | 29938560 | 30331776 | 30331776 | 3538944 | 26792832 | 0 |
| multi_qkv_static_3track_global_30m_seed1 | multi_qkv_static_3track_global | 28611456 | 29004672 | 29004672 | 2211840 | 26792832 | -1327104 |
| multi_qkv_train_rotation_3track_global_30m_seed1 | multi_qkv_train_rotation_3track_global | 28611456 | 29004672 | 29004672 | 2211840 | 26792832 | -1327104 |
| multi_qkv_position_rotation_3track_global_30m_seed1 | multi_qkv_position_rotation_3track_global | 28611456 | 29004672 | 29004672 | 2211840 | 26792832 | -1327104 |

The negative delta is expected because Q/K/V banks are globally shared across layers. B and C should be interpreted primarily against A/static-global.

`inspect_model_config.py` reports trainable parameters, attention projection
parameters, and non-attention parameters for the canonical configs.

## Full-Run Status

Full 3000-step E002 runs were not executed during implementation. Manual execution is required.

## Implementation QC

```text
uv sync: passed
uv run ruff check .: passed
uv run pytest: 162 passed, 1 skipped
uv run pytest --run-integration: 163 passed
uv run scripts/validate_experiment.py --id E002_multitrack_qkv_shift_register: ok=True, config_count=11, runnable_config_count=5, unimplemented_config_count=6
uv run scripts/verify_data.py --data_root data/fineweb_edu_100m --manifest data/fineweb_edu_100m/manifest.json --verify_hashes: passed
uv run scripts/verify_data.py --config configs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1.yaml --verify_hashes: passed
uv run attn-queue doctor --experiment E002_multitrack_qkv_shift_register: passed
```
