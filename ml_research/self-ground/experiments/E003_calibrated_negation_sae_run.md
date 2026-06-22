# E003 Calibrated Negation SAE Run

## Goal

E003 tests whether the prior E002 failure was primarily caused by an
uncalibrated Phase 3 task suite. It builds a larger candidate task bank,
baseline-calibrates tasks before any decoded SAE intervention rows, and reruns
the same decoded SAE evaluation against the calibrated task source.

## Fixed Configuration

- model: `EleutherAI/pythia-70m-deduped`
- hook point: `blocks.2.hook_resid_post`
- SAE release: `pythia-70m-deduped-res-sm`
- SAE id: `blocks.2.hook_resid_post`
- local execution stack: TransformerLens
- SAE runtime: SAELens
- required families:
  - `sentiment_negation`
  - `property_negation`
  - `state_negation`
- task source: calibrated task bank file
- minimum calibrated tasks per family: `10`
- baseline margin threshold: `0.1`
- SAE ranking top-k output: `50`
- evaluated top-k features: `5`
- baseline mode: `top-vs-random-density-and-bottom-active`
- random seeds: `7,11,13`
- operation: `ablate`
- patch mode: `delta`
- serious device: `cuda`

## Command

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
  --out-root runs
```

## Expected Artifacts

```text
data/phase3_task_bank/pythia70m_negation_candidate_bank.json
data/phase3_task_bank/pythia70m_negation_candidate_tasks.jsonl
data/phase3_task_bank/pythia70m_negation_candidate_rejections.jsonl

runs/e003_task_bank_calibration_pythia70m_margin0p1_min10/
  config.json
  candidate_baseline_scores.jsonl
  calibrated_behavioral_tasks.jsonl
  calibrated_excluded_behavioral_tasks.jsonl
  calibration_summary.json
  calibration_by_family.csv
  calibration_by_template.csv
  README.md

runs/e003_real_sae_ranking_pythia70m_l2_calibrated_pf10_top50/
  config.json
  task_source.json
  activation_metadata.json
  feature_rankings.csv
  top_examples.jsonl
  README.md

runs/e003_negation_eval_pythia70m_l2_calibrated_pf10_top5_density/
  config.json
  task_source.json
  source_calibration_summary.json
  source_calibrated_excluded_behavioral_tasks.jsonl
  compatibility.json
  behavioral_tasks.jsonl
  behavioral_task_validation.json
  baseline_task_scores.jsonl
  behavioral_intervention_results.jsonl
  behavioral_summary.csv
  mechanism_report.json
  mechanism_report.md
  inspection_summary.json

runs/diagnostics/e003_vs_e002_comparison/
  comparison.json
  comparison.csv
  family_comparison.csv
  claim_delta.md
```

## Claim Interpretation

Candidate evidence is possible only if `mechanism_report.json` says
`candidate_evidence` or stronger. Any positive claim is conditional on the
baseline-calibrated task bank and the current custom token-contrast evaluator.

Unsupported claims remain:

- broad negation mechanism discovery
- upstream SAEBench/RAVEL benchmark evidence
- model-general or layer-general conclusions
- evidence from dropped or uncalibrated tasks

## 2026-06-21 Execution Result

E003 ran on CUDA in this environment.

Task-bank generation:

- candidates requested: 80 per family;
- accepted/token-valid candidates: 240 total;
- accepted by family:
  `property_negation=80`, `sentiment_negation=80`, `state_negation=80`.

Baseline-only calibration:

- artifact:
  `runs/e003_task_bank_calibration_pythia70m_margin0p1_min10/calibration_summary.json`
- min baseline margin: `0.1`;
- min per family: `10`;
- passes minimum: `true`;
- kept tasks:
  `property_negation=10`, `sentiment_negation=36`, `state_negation=23`;
- excluded by reason:
  `baseline_wrong_direction=161`, `baseline_margin_below_threshold=10`.

Evaluation:

- artifact:
  `runs/e003_negation_eval_pythia70m_l2_calibrated_pf10_top5_density/mechanism_report.json`
- run classification: `serious_gpu_evidence_run`;
- behavioral rows: 552;
- skipped rows: 0;
- claim status: `insufficient_evidence`;
- top target delta: `0.6277369900026183`;
- top matched-control delta: `0.7188387469968934`;
- specificity gap: `-0.09110175699427508`.

Comparison against E002:

- artifact: `runs/diagnostics/e003_vs_e002_comparison/comparison.json`;
- interpretation:
  `calibration_fixed_task_suite_but_feature_specificity_still_failed`;
- E002 baseline pass rate: `0.23333333333333334`;
- E003 baseline pass rate: `1.0`;
- E002 specificity gap: `-0.02048155466715495`;
- E003 specificity gap: `-0.09110175699427508`.

Conclusion: E003 fixed the task-suite calibration blocker, including
`property_negation`, but it did not produce candidate evidence. The selected
SAE feature set moved target prompts substantially, but it moved matched
controls more.
