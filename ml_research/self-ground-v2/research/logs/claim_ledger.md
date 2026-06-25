# Claim Ledger

### C001 - SELF-GROUND can execute real model, residual, and decoded-SAE paths

```yaml
claim_id: C001
title: SELF-GROUND can execute real model, residual, and decoded-SAE paths
status: single_run_evidence
scope: Pythia-70M/Pythia-70M-deduped SELF-GROUND Phase 1-2 path validation
allowed:
  - "engineering verified"
  - "single-run"
  - "real path"
forbidden:
  - "is the mechanism"
  - "the negation feature"
  - "proves that"
  - "causally demonstrates"
required_caveats:
  - "single run"
debt_flags:
  - singleton_seed
linked_experiments:
  - E001
linked_runs:
  - runs/check_real_model.json
  - runs/test_real_residual_intervention
  - runs/test_real_sae_intervention
linked_decisions:
  - D001
copilot_session_id: null
```

Evidence:
- Copied from `self-ground-research/work_logs/claim_ledger.md` Claims 1.1-2.3.
- Real TransformerLens path, residual intervention path, and decoded SAE path
  produced finite outputs.

Contradicting evidence:
- None for path execution.

Required next evidence:
- Do not promote beyond path/single-run language without controls.

Notes:
This backfills real SELF-GROUND Phase 1-2 history into PRD Section 10 format.

### C002 - E002 decoded SAE top features are not negation-specific under matched controls

```yaml
claim_id: C002
title: E002 decoded SAE top features are not negation-specific under matched controls
status: failed_or_weakened
scope: Pythia-70M-deduped, blocks.2.hook_resid_post, E002 uncalibrated task setup
allowed:
  - "insufficient evidence"
  - "matched controls moved more"
  - "weakened"
forbidden:
  - "candidate evidence"
  - "proves that"
  - "is the mechanism"
  - "identifies the mechanism"
required_caveats:
  - "matched-control"
debt_flags:
  - missing_baseline_calibration
  - missing_empirical_null
linked_experiments:
  - E002
linked_runs:
  - runs/e002_negation_ravel_eval_pythia70m_deduped_l2_pf10_top5_density
linked_decisions:
  - D001
copilot_session_id: null
```

Evidence:
- `self-ground/work_logs/claim_ledger.md`: E002 target delta
  `0.03419952392578125`, matched-control delta `0.0546810785929362`,
  specificity gap `-0.02048155466715495`, baseline pass rate below candidate
  threshold.

Contradicting evidence:
- None supporting candidate status; E002 weakens the feature-specific claim.

Required next evidence:
- Fix task calibration and feature specificity without lowering thresholds.

### C003 - E003 calibrated task bank fixes baseline coverage but not feature specificity

```yaml
claim_id: C003
title: E003 calibrated task bank fixes baseline coverage but not feature specificity
status: failed_or_weakened
scope: Pythia-70M-deduped, blocks.2.hook_resid_post, calibrated E003 task bank
allowed:
  - "baseline calibrated"
  - "insufficient evidence"
  - "not negation-specific"
forbidden:
  - "candidate evidence"
  - "causally demonstrates"
  - "isolates the mechanism"
required_caveats:
  - "matched-control"
debt_flags:
  - missing_empirical_null
  - missing_paired_statistic
linked_experiments:
  - E003
linked_runs:
  - runs/e003_task_bank_calibration_pythia70m_margin0p1_min10
  - runs/e003_negation_eval_pythia70m_l2_calibrated_pf10_top5_density
linked_decisions:
  - D001
copilot_session_id: null
```

Evidence:
- `self-ground/work_logs/claim_ledger.md`: calibrated kept tasks
  `property_negation=10`, `sentiment_negation=36`, `state_negation=23`;
  baseline pass rate `1.0`; top target delta `0.6277369900026183`;
  matched-control delta `0.7188387469968934`; specificity gap
  `-0.09110175699427508`.

Contradicting evidence:
- Matched controls moved more than target prompts.

Required next evidence:
- Diagnose specificity failure; do not drop required families.

### C004 - E004 specificity rescue matrix produced no candidate cells

```yaml
claim_id: C004
title: E004 specificity rescue matrix produced no candidate cells
status: failed_or_weakened
scope: Pythia-70M-deduped, block 1/2/3 rescue matrix, multi-control evaluation
allowed:
  - "no candidate cells"
  - "insufficient evidence"
  - "fails at least one control"
forbidden:
  - "identifies the mechanism"
  - "the negation mechanism"
  - "candidate evidence"
  - "strong candidate evidence"
  - "broad negation mechanism discovery"
  - "monosemantic feature claims"
required_caveats:
  - "multi-control"
debt_flags:
  - missing_empirical_null
  - missing_paired_statistic
linked_experiments:
  - E004
linked_runs:
  - runs/e004_specificity_rescue_matrix
  - runs/e004_specificity_rescue_matrix/eval/block1_ensemble_specificity_ablate_amplify_multi
linked_decisions:
  - D001
copilot_session_id: null
```

Evidence:
- `self-ground/work_logs/claim_ledger.md`: 15 attempted cells, 15 completed,
  0 candidate cells. Best aggregate specificity gap was positive
  (`0.13617621988490008`) but best run failed at least one configured control
  suite and at least one required family.

Contradicting evidence:
- The best aggregate run still had multi-control minimum gap
  `-0.01942424497742584` and family minimum gap `-0.0900231236996858`.

Required next evidence:
- Decide whether to retire this SAE/hook search or redesign the objective.
