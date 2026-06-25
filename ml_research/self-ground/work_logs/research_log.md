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

## 2026-06-21: E002 Task Calibration And Feature-Selection Diagnosis

Artifact-only calibration analysis:

```bash
uv run python scripts/analyze_task_calibration.py \
  --run-dir runs/e002_negation_ravel_eval_pythia70m_deduped_l2_pf10_top5_density \
  --out runs/diagnostics/e002_task_calibration
```

Result:

- 30 baseline rows,
- intended-direction pass count: 7,
- intended-direction pass rate: `0.23333333333333334`,
- wrong-direction task count: 23,
- tiny-margin task count at 0.1: 4,
- `property_negation`: 0 / 10 intended-direction pass,
- `sentiment_negation`: 5 / 10 intended-direction pass,
- `state_negation`: 2 / 10 intended-direction pass.

Artifact-only feature-selection analysis:

```bash
uv run python scripts/analyze_feature_selection.py \
  --ranking-dir runs/e002_real_sae_ranking_pythia70m_deduped_l2_pf10_top50 \
  --eval-dir runs/e002_negation_ravel_eval_pythia70m_deduped_l2_pf10_top5_density \
  --out runs/diagnostics/e002_feature_selection
```

Result labels:

- `target_effect_present_but_not_specific`
- `control_effect_dominates`

Bounded calibrated variants:

```bash
uv run python scripts/run_negation_ravel_eval.py ... \
  --out runs/e002_calibrated_intended_direction_top_abs \
  --task-calibration-mode baseline-intended-direction \
  --feature-selection-mode top-absolute

uv run python scripts/run_negation_ravel_eval.py ... \
  --out runs/e002_calibrated_margin_top_abs \
  --task-calibration-mode baseline-margin \
  --min-baseline-margin 0.1 \
  --feature-selection-mode top-absolute

uv run python scripts/run_negation_ravel_eval.py ... \
  --out runs/e002_calibrated_intended_direction_top_positive \
  --task-calibration-mode baseline-intended-direction \
  --feature-selection-mode top-positive
```

All three variants wrote blocker artifacts and no intervention rows because
baseline-only calibration failed required family coverage with
`--allow-family-drop false`.

Comparison artifact:

```text
runs/diagnostics/e002_variant_comparison/comparison.json
```

Interpretation:

The current failure is primarily task calibration: property-negation token
contrasts are not baseline-calibrated for Pythia-70M-deduped. The uncalibrated
feature set also lacks specificity because matched controls move more than
target prompts. No variant reached candidate evidence.

## 2026-06-21: SAEBench/RAVEL Boundary Decision

Ran:

```bash
uv run python scripts/probe_saebench_ravel_bridge.py \
  --out runs/tooling_spikes/saebench_ravel_bridge
```

Result: `not_installed`.

Decision recorded:

```text
docs/decision_log/D008_saebench_ravel_eval_boundary.md
```

No upstream SAEBench/RAVEL eval ran. Current results remain RAVEL-shaped custom
token-contrast evaluation over real decoded SAE interventions.

## 2026-06-21: E003 Calibrated Task Bank Run

Built and ran the calibrated E003 task-bank pipeline:

```bash
uv run python scripts/run_e003_calibrated_negation_sae.py \
  --device cuda \
  --task-bank data/phase3_task_bank/pythia70m_negation_candidate_bank.json \
  --per-family-candidates 80 \
  --min-calibrated-per-family 10 \
  --min-baseline-margin 0.1 \
  --ranking-top-k 50 \
  --eval-top-k 5 \
  --operations ablate \
  --random-seeds 7,11,13 \
  --out-root runs \
  --force
```

Task bank:

- artifact: `data/phase3_task_bank/pythia70m_negation_candidate_bank.json`
- accepted/token-valid candidates: 240 total, 80 per required family
- rejected tokenization candidates: 0

Calibration:

- artifact:
  `runs/e003_task_bank_calibration_pythia70m_margin0p1_min10/calibration_summary.json`
- passes minimum: true
- kept tasks: `property_negation=10`, `sentiment_negation=36`,
  `state_negation=23`
- excluded by reason: `baseline_wrong_direction=161`,
  `baseline_margin_below_threshold=10`

Evaluation:

- ranking:
  `runs/e003_real_sae_ranking_pythia70m_l2_calibrated_pf10_top50`
- evaluation:
  `runs/e003_negation_eval_pythia70m_l2_calibrated_pf10_top5_density`
- comparison:
  `runs/diagnostics/e003_vs_e002_comparison/comparison.json`
- behavioral rows: 552
- skipped rows: 0
- claim status: `insufficient_evidence`
- top target delta: `0.6277369900026183`
- top control delta: `0.7188387469968934`
- specificity gap: `-0.09110175699427508`

Interpretation:

E003 repaired the baseline task-suite failure, including `property_negation`.
The run still does not support candidate evidence because matched-control
movement exceeds target-prompt movement. The current blocker is no longer
family coverage; it is negation-specific feature/control separation under the
current task/evaluator setup.

## 2026-06-24: E004 Specificity Rescue Matrix

Ran the artifact-only E003 specificity diagnosis:

```bash
uv run python scripts/diagnose_e003_specificity_failure.py \
  --run-dir runs/e003_negation_eval_pythia70m_l2_calibrated_pf10_top5_density \
  --ranking-dir runs/e003_real_sae_ranking_pythia70m_l2_calibrated_pf10_top50 \
  --calibration-dir runs/e003_task_bank_calibration_pythia70m_margin0p1_min10 \
  --out runs/diagnostics/e003_specificity_failure
```

Diagnosis labels:

- `control_dominates_globally`
- `control_dominates_specific_families`
- `possible_control_prompt_confound`
- `specificity_failure_concentrated_in_template`
- `target_effect_present_but_nonspecific`
- `true_negative_for_current_layer_feature_set`

Ran the bounded E004 matrix on CUDA:

```bash
uv run python scripts/run_e004_specificity_rescue_matrix.py \
  --device cuda \
  --task-file runs/e003_task_bank_calibration_pythia70m_margin0p1_min10/calibrated_behavioral_tasks.jsonl \
  --task-bank-calibration-dir runs/e003_task_bank_calibration_pythia70m_margin0p1_min10 \
  --layers blocks.1.hook_resid_post,blocks.2.hook_resid_post,blocks.3.hook_resid_post \
  --feature-selection-modes top-absolute,top-target-control-gap,top-family-consistent-gap,top-low-control-activation,ensemble-specificity \
  --operations ablate,amplify \
  --control-suite multi_control \
  --ranking-top-k 100 \
  --eval-top-k 5 \
  --min-family-consistency 2 \
  --random-seeds 7,11,13 \
  --out-root runs/e004_specificity_rescue_matrix
```

Matrix result:

- attempted cells: 15
- completed cells: 15
- blocked cells: 0
- candidate cells: 0
- best aggregate run:
  `runs/e004_specificity_rescue_matrix/eval/block1_ensemble_specificity_ablate_amplify_multi`
- best aggregate claim status: `insufficient_evidence`
- best aggregate target delta: `0.8960492369057476`
- best aggregate control delta: `0.7598730170208475`
- best aggregate specificity gap: `0.13617621988490008`
- best aggregate top/control ratio: `1.179209179474212`
- multi-control minimum gap: `-0.01942424497742584`
- family minimum gap: `-0.0900231236996858`

Interpretation:

E004 improved aggregate specificity over E003 in block 1, but no cell reached
candidate evidence. The best aggregate run still failed `multi_control`
because at least `hard_negative_control` and
`matched_non_negation_current` failed, and one required family remained
negative. The current model/SAE/task setup should be treated as unsupported
for the negation-specific SAE feature-set claim.
