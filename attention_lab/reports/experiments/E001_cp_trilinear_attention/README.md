# E001 CP Trilinear Attention

This report directory is reserved for the planned CP-bilinear and CP-trilinear
attention experiment. No CP results are claimed yet.

## Paths

- Plan: `docs/experiments/E001_cp_trilinear_attention_plan.md`
- Configs: `configs/experiments/E001_cp_trilinear_attention/`
- Runs: `runs/experiments/E001_cp_trilinear_attention/`
- Comparison template: `comparison_template.json`
- Results template: `results_template.md`

## Required Completion Gate

Before any result is reported, every completed run must pass:

```text
verify_run.py --expect-complete-training --expect-sample --expect-eval-loss --expect-hellaswag --expect-data-manifest
```
