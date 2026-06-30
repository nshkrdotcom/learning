# E002 Multitrack QKV Shift Register

Status: implementation prepared; no full-run result is claimed until manual full runs complete and verify.

This directory holds E002 queue exports, run indexes, decision logs, implementation notes, templates, and future verified
results.

## Canonical Initial Runs

| Run | Role |
| --- | --- |
| `standard_refactor_control_30m_seed1` | Shared-path standard control |
| `multi_qkv_static_3track_global_30m_seed1` | A: static global 3-track cycle |
| `multi_qkv_train_rotation_3track_global_30m_seed1` | B: train-time phase rotation, eval freeze |
| `multi_qkv_position_rotation_3track_global_30m_seed1` | C: position-clock routing at train/eval/generate |

## Required Artifacts Per Full Run

- checkpoint
- metrics
- `evals/val_loss.json`
- `evals/hellaswag.json`
- `evals/run_summary.json`
- `evals/attention_diagnostics.jsonl` for Multi-QKV runs
- `evals/qkv_track_destructive_test.json` for Multi-QKV runs

## Evidence Rule

Validation loss is not interpretable without mechanism diagnostics. A Multi-QKV run with missing or degenerate diagnostics is
`insufficient_evidence`.

Current status:

```text
implementation: first-build A/B/C code and configs prepared
tiny/integration evidence: covered by pytest, not scientific evidence
full 3000-step runs: not executed by the implementation agent
verified E002 result claims: none yet
```

Manual full-run scripts live under:

```text
scripts/experiments/E002_multitrack_qkv_shift_register/
```

Do not add comparison claims until actual run artifacts pass `verify_run.py`, `eval_loss.py`, `eval_generate.py`,
`eval_hellaswag.py`, `summarize_run.py`, `qkv_track_activity`, and the destructive route test.
