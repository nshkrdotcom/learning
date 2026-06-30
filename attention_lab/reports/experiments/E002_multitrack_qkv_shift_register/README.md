# E002 Multi-QKV Shift Register Reports

This directory holds E002 queue exports, run indexes, decision logs, implementation notes, and future verified results.

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

Do not add comparison claims until actual run artifacts pass `verify_run.py`, `eval_loss.py`, `eval_generate.py`, `eval_hellaswag.py`, and `summarize_run.py`.
