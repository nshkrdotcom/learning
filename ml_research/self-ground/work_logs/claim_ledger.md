# Claim Ledger

## Claim: Negation-Scope SAE Feature Set Candidate

- current status: `insufficient_evidence`
- latest serious run: `runs/e002_negation_ravel_eval_pythia70m_deduped_l2_pf10_top5_density`
- latest report: `runs/e002_negation_ravel_eval_pythia70m_deduped_l2_pf10_top5_density/mechanism_report.json`
- latest inspection: `runs/e002_negation_ravel_eval_pythia70m_deduped_l2_pf10_top5_density/inspection_summary.json`

The latest serious run is artifact-backed and uses the real TransformerLens +
SAELens decoded SAE path, but it does not support candidate evidence:

- target delta is nonzero: `0.03419952392578125`,
- matched-control delta is larger: `0.0546810785929362`,
- specificity gap is negative: `-0.02048155466715495`,
- baseline intended-direction pass rate is below candidate threshold,
- density matching is approximate and relaxed.

No broad mechanism discovery, monosemanticity, broad behavioral understanding,
or genuine introspection claim is supported.

## Path Status

- diagnostic path completed: `runs/diagnostic_negation_ravel_eval_density_matched`
- serious E002 completed: `runs/e002_negation_ravel_eval_pythia70m_deduped_l2_pf10_top5_density`
- serious E002 blocked: no
- large CPU fallback completed: not needed because CUDA was available
- zero-effect diagnosis completed: `runs/diagnostics/zero_effect_diagnostic_negation_ravel_eval_density_matched`
- patch sanity check completed:
  - old diagnostic ranking: `runs/diagnostics/check_decoded_sae_patch_nonzero`
  - E002 ranking: `runs/diagnostics/check_decoded_sae_patch_nonzero_e002`

## Next Promotion Requirement

Next promotion requires fixing task calibration and/or feature selection so
target-prompt movement exceeds matched-control movement under the same
artifact-backed E002 conditions.

## 2026-06-21 Calibration Follow-Up

- calibration analysis:
  `runs/diagnostics/e002_task_calibration/calibration_summary.json`
- feature-selection analysis:
  `runs/diagnostics/e002_feature_selection/feature_specificity_diagnosis.json`
- variant comparison:
  `runs/diagnostics/e002_variant_comparison/comparison.json`

Result:

- original E002 remains `insufficient_evidence`;
- baseline intended-direction pass rate is `0.23333333333333334`;
- `property_negation` retains 0 / 10 tasks under intended-direction
  calibration;
- `state_negation` retains 2 / 10 tasks under intended-direction calibration;
- all three bounded calibrated variants are `blocked` by
  `task_calibration_failed`;
- top-positive feature selection did not get evaluated because calibration
  failed before intervention rows.

Strongest supported claim:

The real decoded SAE intervention path moves logits, but the current
Pythia-70M-deduped E002 task/feature setup does not support a negation-specific
SAE feature-set claim. The immediate bottleneck is baseline task calibration,
with feature specificity also unresolved in the uncalibrated run.

Unsupported:

- candidate evidence for a negation-scope SAE feature set,
- broad negation mechanism discovery,
- upstream SAEBench/RAVEL benchmark evidence.

## 2026-06-21 E003 Calibrated Task Bank Result

- task bank:
  `data/phase3_task_bank/pythia70m_negation_candidate_bank.json`
- calibration:
  `runs/e003_task_bank_calibration_pythia70m_margin0p1_min10/calibration_summary.json`
- evaluation:
  `runs/e003_negation_eval_pythia70m_l2_calibrated_pf10_top5_density/mechanism_report.json`
- inspection:
  `runs/e003_negation_eval_pythia70m_l2_calibrated_pf10_top5_density/inspection_summary.json`
- comparison:
  `runs/diagnostics/e003_vs_e002_comparison/comparison.json`

E003 status: `insufficient_evidence`.

Task-suite result:

- candidate tasks: 240 total, 80 per required family;
- token-valid candidates: 240;
- calibrated kept tasks:
  `property_negation=10`, `sentiment_negation=36`, `state_negation=23`;
- baseline intended-direction pass rate in the evaluated run: `1.0`.

Evidence result:

- behavioral rows: 552;
- skipped rows: 0;
- top target delta: `0.6277369900026183`;
- top matched-control delta: `0.7188387469968934`;
- specificity gap: `-0.09110175699427508`;
- interpretation category:
  `calibration_fixed_task_suite_but_feature_specificity_still_failed`.

Strongest supported claim:

The calibrated E003 task bank fixes the baseline task-suite coverage blocker
for Pythia-70M-deduped, including `property_negation`. The selected SAE feature
set moves logits under real decoded intervention, but the effect is not
negation-specific under the current matched-control evaluation.

Unsupported:

- candidate evidence for a negation-scope SAE feature set;
- broad negation mechanism discovery;
- upstream SAEBench/RAVEL benchmark evidence;
- model-general, layer-general, or task-general conclusions.

Next action:

Diagnose why the top calibrated SAE feature set moves matched non-negation
controls more than target prompts. Do not change evidence thresholds or drop
required families.
