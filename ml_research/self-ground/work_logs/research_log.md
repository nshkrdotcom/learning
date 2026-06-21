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

## 2026-06-21: E002 Capability And Serious GPU Run

Capability check:

```bash
uv run python scripts/check_run_capability.py \
  --out runs/capability_check \
  --model EleutherAI/pythia-70m-deduped \
  --sae-release pythia-70m-deduped-res-sm \
  --sae-id blocks.2.hook_resid_post
```

Result:

- CUDA available: `true`
- device: `NVIDIA GeForce RTX 5060 Ti`
- TransformerLens importable: `true`
- SAELens importable: `true`
- can attempt E002 GPU: `true`

Ran:

```bash
uv run python scripts/run_e002_negation_sae_density_matched.py \
  --device cuda \
  --ranking-per-family 10 \
  --eval-per-family 10 \
  --ranking-top-k 50 \
  --eval-top-k 5 \
  --operations ablate \
  --random-seeds 7,11,13 \
  --out-root runs \
  --allow-cpu-serious-run false
```

Artifacts:

- ranking: `runs/e002_real_sae_ranking_pythia70m_deduped_l2_pf10_top50`
- evaluation: `runs/e002_negation_ravel_eval_pythia70m_deduped_l2_pf10_top5_density`
- inspection: `runs/e002_negation_ravel_eval_pythia70m_deduped_l2_pf10_top5_density/inspection_summary.json`

Result:

- run classification: `serious_gpu_evidence_run`
- valid tasks: 30
- feature sets: 8 (`top`, 3 random, 3 density-matched, `bottom_active`)
- behavioral rows: 240
- skipped rows: 0
- claim status: `insufficient_evidence`
- top target delta: `0.03419952392578125`
- top control delta: `0.0546810785929362`
- specificity gap: `-0.02048155466715495`

Interpretation: the real decoded SAE intervention moved logits under the serious
setting, but matched-control movement exceeded target movement for the top
feature set and baseline calibration remained below threshold. The result is an
empirical effect, not candidate evidence.

## 2026-06-21: Zero-Effect And Patch Diagnostics

Diagnosed the earlier all-zero diagnostic run:

```bash
uv run python scripts/diagnose_zero_effect_run.py \
  --run-dir runs/diagnostic_negation_ravel_eval_density_matched \
  --ranking-dir runs/test_real_sae_ranking \
  --out runs/diagnostics/zero_effect_diagnostic_negation_ravel_eval_density_matched
```

Labels:

- `all_row_deltas_zero`
- `decoded_delta_norm_zero_or_missing`
- `task_baseline_not_calibrated`

Patch sanity on the old diagnostic ranking:

- artifact: `runs/diagnostics/check_decoded_sae_patch_nonzero/patch_check.json`
- selected features inactive on `The movie was not`
- `feature_delta_l1=0.0`
- `decoded_delta_norm=0.0`
- `max_abs_logit_delta=0.0`

Patch sanity on the E002 ranking:

- artifact: `runs/diagnostics/check_decoded_sae_patch_nonzero_e002/patch_check.json`
- selected features include an active feature on `The movie was not`
- `feature_delta_l1=0.6299780011177063`
- `decoded_delta_norm=0.6299779415130615`
- `max_abs_logit_delta=0.2633657455444336`

Interpretation: the prior all-zero diagnostic was explained by inactive selected
features and poor task calibration, not by a globally broken decoded patch path.
The E002-selected feature set produces a nonzero decoded patch and changes
logits.

## 2026-06-21: Optional Ablate+Amplify Diagnostic

Ran a small CPU diagnostic with both ablation and amplification on the E002
ranking:

```bash
uv run python scripts/run_negation_ravel_eval.py \
  --ranking-dir runs/e002_real_sae_ranking_pythia70m_deduped_l2_pf10_top50 \
  --out runs/diagnostic_negation_ravel_eval_density_matched_ablate_amplify \
  --model EleutherAI/pythia-70m-deduped \
  --hook-point blocks.2.hook_resid_post \
  --sae-release pythia-70m-deduped-res-sm \
  --sae-id blocks.2.hook_resid_post \
  --per-family 2 \
  --top-k-features 2 \
  --baseline-mode top-vs-density-matched-multiseed \
  --random-seeds 7,11,13 \
  --operations ablate,amplify \
  --amplify-factors 2.0 \
  --patch-mode delta \
  --device cpu
```

Result: `insufficient_evidence`, with tiny nonzero target movement
(`top_target_delta=0.00047969818115234375`) at diagnostic scale.
