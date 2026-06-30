# Architecture Variant Checklist

Use this checklist for every new attention architecture. Baseline files must not be
edited unless the experiment explicitly includes a standard-refactor control.

1. Add the architecture module under `src/attention_lab/models/attention/`.
2. Add a registry entry in `src/attention_lab/models/attention/registry.py`.
3. Add config validation support for the new `attention_type`.
4. Add unit tests for shape, causal masking, and gradient flow.
5. Add a lambda=0 or equivalent identity/wiring control when applicable.
6. Add parameter-count inspection coverage.
7. Add mechanism diagnostics.
8. Add experiment configs under `configs/experiments/<EXPERIMENT_ID>/`.
9. Add or update the experiment plan.
10. Train the candidate under the same manifest and token budget.
11. Run `eval_loss.py`, `eval_generate.py`, and bounded `eval_hellaswag.py`.
12. Run `summarize_run.py`.
13. Run `verify_run.py` with `--expect-data-manifest`.
14. Run `compare_runs.py`.
15. Update the experiment report.
16. State the claim boundary and negative-result interpretation.

The standard-attention baseline path must remain intact throughout the experiment.
