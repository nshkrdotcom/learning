# E002 Real Negation SAE Density-Matched Run

## Objective

Run the first serious artifact-backed SELF-GROUND negation SAE experiment with
semantic SAE compatibility, activation-density-matched controls, real decoded
SAE intervention, and conservative local claim reporting.

This is not a broad sweep. It is one inspectable run.

## Stack

- model: `EleutherAI/pythia-70m-deduped`
- hook point: `blocks.2.hook_resid_post`
- SAE release: `pythia-70m-deduped-res-sm`
- SAE id: `blocks.2.hook_resid_post`
- execution: TransformerLens
- SAE runtime: SAELens
- audit: SELF-GROUND local artifacts

## Task Families

- `sentiment_negation`
- `property_negation`
- `state_negation`

## Serious GPU Ranking Command

```bash
uv run python scripts/run_real_activation_ranking.py \
  --model EleutherAI/pythia-70m-deduped \
  --hook-point blocks.2.hook_resid_post \
  --feature-source sae \
  --sae-release pythia-70m-deduped-res-sm \
  --sae-id blocks.2.hook_resid_post \
  --device cuda \
  --per-family 10 \
  --top-k-features 50 \
  --out runs/real_sae_ranking_pythia70m_deduped_l2_pf10
```

## Serious GPU Evaluation Command

```bash
uv run python scripts/run_negation_ravel_eval.py \
  --ranking-dir runs/real_sae_ranking_pythia70m_deduped_l2_pf10 \
  --out runs/negation_ravel_eval_pythia70m_deduped_l2_pf10_top5_density \
  --model EleutherAI/pythia-70m-deduped \
  --hook-point blocks.2.hook_resid_post \
  --sae-release pythia-70m-deduped-res-sm \
  --sae-id blocks.2.hook_resid_post \
  --per-family 10 \
  --top-k-features 5 \
  --baseline-mode top-vs-random-density-and-bottom-active \
  --random-seeds 7,11,13 \
  --operations ablate \
  --patch-mode delta \
  --device cuda
```

Amplification can be added only if runtime permits:

```bash
--operations ablate,amplify --amplify-factors 2.0
```

## CPU Diagnostic Path

This command is diagnostic-only by scale and device. It checks that the local
pipeline works; it is not the serious evidence run.

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

Inspect it with:

```bash
uv run python scripts/inspect_claim_run.py \
  --run-dir runs/diagnostic_negation_ravel_eval_density_matched
```

## Expected Artifacts

- `config.json`
- `behavioral_tasks.jsonl`
- `behavioral_task_validation.json`
- `excluded_behavioral_tasks.jsonl`
- `compatibility.json`
- `feature_sets.json`
- `baseline_task_scores.jsonl`
- `baseline_task_summary.csv`
- `baseline_validation.json`
- `behavioral_intervention_results.jsonl`
- `behavioral_summary.csv`
- `skipped_behavioral_rows.json`
- `mechanism_report.json`
- `mechanism_report.md`
- `README.md`

## Claim-Status Expectations

`candidate_evidence` is possible only if all candidate thresholds pass in the
artifact-backed mechanism report.

`strong_candidate_evidence` is impossible unless all strong thresholds pass,
including sufficient task count, multiple control seeds, density-matched control
rows, both operation coverage if configured, low norm drift, and zero skipped
rows.

Failure outcomes must be recorded honestly:

- model load/resource blocker,
- SAE compatibility blocker,
- missing/underfilled required task family,
- non-finite baseline blocker,
- decoded intervention resource failure,
- insufficient target/control separation,
- density matching approximation or relaxed matching limitation.

## 2026-06-21 Execution Result

The E002 CUDA run completed in this environment.

Capability:

- CUDA available: yes
- device: `NVIDIA GeForce RTX 5060 Ti`
- capability artifact: `runs/capability_check/capability.json`

Artifacts:

- ranking: `runs/e002_real_sae_ranking_pythia70m_deduped_l2_pf10_top50`
- evaluation: `runs/e002_negation_ravel_eval_pythia70m_deduped_l2_pf10_top5_density`
- inspection: `runs/e002_negation_ravel_eval_pythia70m_deduped_l2_pf10_top5_density/inspection_summary.json`

Result:

- run classification: `serious_gpu_evidence_run`
- valid tasks: 30
- behavioral rows: 240
- skipped rows: 0
- claim status: `insufficient_evidence`
- top target delta: `0.03419952392578125`
- top control delta: `0.0546810785929362`
- specificity gap: `-0.02048155466715495`

Interpretation:

The decoded SAE intervention moved logits under the serious setting, so the
pipeline is not globally no-op. The result does not support candidate evidence
because matched-control movement exceeded target movement and baseline
calibration remained below threshold.

## 2026-06-21 Calibration And Feature-Selection Follow-Up

Baseline calibration analysis:

- artifact: `runs/diagnostics/e002_task_calibration/calibration_summary.json`
- intended-direction pass count: 7 / 30
- intended-direction pass rate: `0.23333333333333334`
- `property_negation`: 0 / 10 pass
- `sentiment_negation`: 5 / 10 pass
- `state_negation`: 2 / 10 pass

Feature-selection analysis:

- artifact:
  `runs/diagnostics/e002_feature_selection/feature_specificity_diagnosis.json`
- diagnosis labels:
  - `target_effect_present_but_not_specific`
  - `control_effect_dominates`
- top mean abs ranking score: `0.14233589977025984`
- density-control mean abs ranking score: `0.0`

Bounded calibrated variants:

```text
runs/e002_calibrated_intended_direction_top_abs
runs/e002_calibrated_margin_top_abs
runs/e002_calibrated_intended_direction_top_positive
```

All three variants blocked before decoded intervention rows because
preregistered baseline-only calibration failed to retain the required family
coverage with `--allow-family-drop false`.

Comparison artifact:

```text
runs/diagnostics/e002_variant_comparison/comparison.json
```

Interpretation:

The original E002 failure is substantially a task-calibration problem: the
current property-negation templates are not baseline-calibrated for
Pythia-70M-deduped under the selected token contrasts. Feature selection also
remains insufficiently specific because the uncalibrated top feature set moves
matched controls more than target prompts. No calibrated variant produced
candidate evidence.

Next action:

Revise or expand the task suite using baseline-only calibration criteria before
running another serious decoded SAE intervention evaluation. Do not weaken
claim thresholds or drop required families to obtain a positive result.

## E003 Follow-Up

E003 implemented that next action with a calibrated task bank:

- experiment doc: `experiments/E003_calibrated_negation_sae_run.md`
- calibration artifact:
  `runs/e003_task_bank_calibration_pythia70m_margin0p1_min10/calibration_summary.json`
- evaluation artifact:
  `runs/e003_negation_eval_pythia70m_l2_calibrated_pf10_top5_density/mechanism_report.json`
- comparison artifact:
  `runs/diagnostics/e003_vs_e002_comparison/comparison.json`

E003 repaired the task-suite calibration failure:

- candidate tasks: 240 total, 80 per family;
- calibrated kept tasks:
  `property_negation=10`, `sentiment_negation=36`, `state_negation=23`;
- evaluation baseline pass rate: `1.0`.

E003 still remained `insufficient_evidence` because the matched-control delta
(`0.7188387469968934`) exceeded the target delta (`0.6277369900026183`), with
a specificity gap of `-0.09110175699427508`.
