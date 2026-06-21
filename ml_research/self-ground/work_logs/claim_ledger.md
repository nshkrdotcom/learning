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
