# E004 Specificity Rescue Matrix

## Goal

Diagnose why E003 moved matched non-negation controls more than target prompts
after task-bank calibration succeeded. E004 asks whether specificity can be
rescued by stricter controls, richer pre-intervention feature selection,
nearby layers, or operation choice.

This is a bounded serious matrix, not a broad sweep.

## Fixed Constants

- model: `EleutherAI/pythia-70m-deduped`
- execution: TransformerLens
- SAE runtime: SAELens
- SAE release base: `pythia-70m-deduped-res-sm`
- task source:
  `runs/e003_task_bank_calibration_pythia70m_margin0p1_min10/calibrated_behavioral_tasks.jsonl`
- calibration dir:
  `runs/e003_task_bank_calibration_pythia70m_margin0p1_min10`
- required families:
  `sentiment_negation`, `property_negation`, `state_negation`
- min calibrated per family: `10`
- baseline mode: `top-vs-random-density-and-bottom-active`
- random seeds: `7,11,13`
- patch mode: `delta`
- device: `cuda`
- control suite: `multi_control`

## Matrix

Layers:

- `blocks.1.hook_resid_post`
- `blocks.2.hook_resid_post`
- `blocks.3.hook_resid_post`

Feature-selection modes:

- `top-absolute`
- `top-target-control-gap`
- `top-family-consistent-gap`
- `top-low-control-activation`
- `ensemble-specificity`

Operations:

- `ablate,amplify`

The matrix attempts 15 cells. A cell can block if a layer-specific SAE is
unavailable, semantic compatibility fails, ranking fails, control-suite
coverage fails, or decoded intervention execution fails. Blocked cells are kept
as artifacts and do not support candidate evidence.

## Command

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

Compare:

```bash
uv run python scripts/compare_e004_matrix.py \
  --matrix-root runs/e004_specificity_rescue_matrix \
  --e003 runs/e003_negation_eval_pythia70m_l2_calibrated_pf10_top5_density \
  --out runs/e004_specificity_rescue_matrix/comparison
```

## Artifact Contract

Each completed eval cell writes the Phase 3 artifact contract plus:

- `control_suite.json`
- `control_task_mapping.jsonl`
- `control_suite_validation.json`
- `selected_feature_rationale.csv`
- `inspection_summary.json`

Matrix-level outputs:

- `matrix_run_summary.json`
- `comparison/matrix_summary.csv`
- `comparison/matrix_summary.json`
- `comparison/best_runs_by_specificity.csv`
- `comparison/best_runs_by_family.csv`
- `comparison/blocked_runs.csv`
- `comparison/claim_adjudication.md`

Forensics:

- `forensics/forensics_summary.md`
- `forensics/task_outlier_table.csv`
- `forensics/family_breakdown.csv`
- `forensics/token_pair_breakdown.csv`
- `forensics/template_breakdown.csv`
- `forensics/control_suite_breakdown.csv`
- `forensics/feature_breakdown.csv`

## Claim Rules

A run cannot be candidate evidence unless its own `mechanism_report.json`
reports `candidate_evidence` or stronger.

For `multi_control`, candidate evidence also requires:

- every configured control suite has positive top specificity;
- every required family has positive specificity;
- no configured control suite beats target movement;
- existing Phase 3 thresholds remain unchanged.

If no cell reaches candidate evidence, the current Pythia-70M-deduped plus
SAELens calibrated negation setup remains unsupported for this SAE/layer search.

## Unsupported Claims

- broad negation mechanism discovery;
- upstream SAEBench/RAVEL benchmark evidence;
- model-general or layer-general conclusions;
- monosemantic feature claims;
- causal mechanism claims beyond the exact decoded SAE/token-contrast cell.

## Observed Result

Artifacts:

- matrix summary:
  `runs/e004_specificity_rescue_matrix/matrix_run_summary.json`
- comparison:
  `runs/e004_specificity_rescue_matrix/comparison/comparison.json`
- adjudication:
  `runs/e004_specificity_rescue_matrix/comparison/claim_adjudication.md`
- forensics:
  `runs/e004_specificity_rescue_matrix/forensics/forensics_summary.md`

Result:

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

E004 improved aggregate target/control specificity over E003 in block 1, but
the stricter `multi_control` and per-family gates did not pass. The current
Pythia-70M-deduped / `pythia-70m-deduped-res-sm` calibrated negation setup
remains unsupported as candidate evidence.
