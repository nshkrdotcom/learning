# Research Log

## 2026-06-21: Library-Backed Diagnostic Run

Ran the real SELF-GROUND Phase 3 RAVEL-shaped token-contrast path:

```bash
uv run python scripts/run_negation_ravel_eval.py \
  --ranking-dir runs/test_real_sae_ranking \
  --out runs/diagnostic_negation_ravel_eval_density_matched \
  --model EleutherAI/pythia-70m-deduped \
  --hook-point blocks.2.hook_resid_post \
  --sae-release pythia-70m-deduped-res-sm \
  --sae-id blocks.2.hook_resid_post \
  --per-family 2 \
  --top-k-features 2 \
  --baseline-mode top-vs-density-matched-multiseed \
  --random-seeds 7,11,13 \
  --operations ablate \
  --patch-mode delta \
  --device cpu
```

Then inspected the artifact-backed claim state:

```bash
uv run python scripts/inspect_claim_run.py \
  --run-dir runs/diagnostic_negation_ravel_eval_density_matched
```

Result:

- real TransformerLens model path completed,
- real SAELens SAE compatibility passed,
- 6 valid tasks across the three required families,
- 3 density-matched control feature sets,
- 24 behavioral rows,
- 0 skipped rows,
- claim status: `insufficient_evidence`.

Reason for conservative status:

- `top_target_delta=0.0`,
- `top_control_delta=0.0`,
- `specificity_gap=0.0`,
- baseline intended-direction pass rate below candidate threshold,
- density matching used per-condition mean approximations,
- density matching required relaxed tolerances,
- no amplification sweep was run.

This run validates the local execution/audit path. It is not serious evidence.
The serious GPU command is specified in
`experiments/E002_real_negation_sae_density_matched_run.md`.

## 2026-06-21: SAEBench/RAVEL Probe

Ran:

```bash
uv run python scripts/probe_saebench_ravel_bridge.py \
  --out runs/tooling_spikes/saebench_ravel_bridge
```

Result: `not_installed`.

The probe attempted `sae_bench`, `saebench`, `sae_bench.evals.ravel`,
`sae_bench.evals.ravel.eval`, and `ravel`. All failed with
`ModuleNotFoundError`. No upstream SAEBench/RAVEL integration is claimed.
