# 0901 Rebuild 02: Model and Registry Implementation

## Required Files

```text
src/attention_lab/models/attention/multi_qkv_common.py
src/attention_lab/models/attention/multi_qkv_static.py
src/attention_lab/models/attention/multi_qkv_train_rotation.py
src/attention_lab/models/attention/multi_qkv_position_rotation.py
src/attention_lab/models/attention/registry.py
src/attention_lab/models/gpt.py
src/attention_lab/training/train.py
src/attention_lab/training/config.py
```

## Shared Bank Contract

The model must create one `MultiQKVSharedBank` and pass that same object to every transformer block for Multi-QKV attention types. A private bank per layer is not acceptable for first-build E002.

The bank owns:

```text
q_proj[0..2]
k_proj[0..2]
v_proj[0..2]
```

Each projection maps `n_embd -> n_embd` and matches ordinary full-dense Q/K/V projection shape.

## Attention Types

```text
multi_qkv_static_3track_global
multi_qkv_train_rotation_3track_global
multi_qkv_position_rotation_3track_global
```

These names are canonical. Old skeleton names remain experimental/unimplemented planning files.

## Threaded Context

`GPT.forward` accepts optional schedule context:

```python
def forward(self, idx, targets=None, *, step=None, positions=None, schedule_mode=None):
    ...
```

`Block.forward` passes `step`, `positions`, `schedule_mode`, and `layer_idx` to attention. Standard and CP modules accept these optional keywords and ignore them.

Training passes the current global optimizer step with `schedule_mode="train"`. Eval loss and HellaSwag pass `schedule_mode="eval"`. Generation passes `schedule_mode="generate"`. B freezes to static depth routing in eval/generation and must not use the training step clock there.

## Registry

`build_attention(config, layer_idx=None, shared_qkv_bank=None)` dispatches standard, CP, and Multi-QKV modules. Multi-QKV modules require `layer_idx` and `shared_qkv_bank`.

## Config Validation

`GPTConfig` and strict YAML validation accept:

```yaml
model:
  qkv_track_count: 3
  qkv_global_bank: true
  qkv_route_formula: layer_mod
```

The three canonical Multi-QKV attention types require `qkv_track_count: 3`, `qkv_global_bank: true`, and the matching `qkv_route_formula`. Legacy `multi_qkv_track_count`/`multi_qkv_global` aliases may be accepted for compatibility, but canonical E002 configs use `qkv_*`.
