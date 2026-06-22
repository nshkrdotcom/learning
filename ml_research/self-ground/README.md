# SELF-GROUND

SELF-GROUND is a negation-scope interpretability experiment harness. The repo
now has a Phase 1 real residual-dimension pipeline, Phase 2 decoded SAE
intervention infrastructure, and Phase 3 token-contrast evaluation reports. It
does not claim complete SELF-GROUND, broad mechanism discovery, broad behavioral
understanding, or genuine model introspection.

SELF-GROUND is not a generic intervention engine. Local execution and patching
use TransformerLens, SAE transforms use SAELens, Phase 3 is moving toward a
RAVEL/SAEBench cause/isolation evaluation shape, and SELF-GROUND owns the
negation task specification plus artifact-backed claim ledger.

The earlier MechanismLab framework extraction attempt is frozen and removed from
the active package. The current priority is one serious, inspectable,
artifact-backed negation SAE run over the existing TransformerLens + SAELens
stack. Framework extraction should not resume until multiple distinct tasks have
run end to end and a shared abstraction would delete code or reduce complexity.

## Phase 1 Recap

Phase 1 implements:

- deterministic negation minimal pairs,
- real TransformerLens activation capture,
- real residual-dimension activation ranking,
- real residual-dimension smoke patching through TransformerLens hooks,
- real logit-contrast deltas after residual smoke patches.

Phase 1 residual outputs are diagnostic only. They do not enter
`candidate_evidence` or `strong_candidate_evidence`, do not claim sparse SAE
mechanisms, and are not paper-facing evidence. Raw residual dimensions are
basis-dependent.

## Phase 2 Goal

Phase 2 adds decoded SAE feature intervention:

```text
real model activations
  -> real SAELens encode
  -> modify selected SAE feature activations
  -> decode back to residual space
  -> patch the real TransformerLens model
  -> rerun logits
  -> write intervention artifacts
```

If no compatible SAE release/id is available, Phase 2 writes a precise compatibility artifact and does not write fabricated intervention rows.

## Phase 3 Goal

Phase 3 evaluates decoded SAE interventions across deterministic
negation-sensitive token-contrast tasks. It validates tokenization, records
baseline task calibration, compares top SAE feature sets against deterministic
control feature sets, scores both target prompts and matched non-negation
control prompts, records intervention telemetry, and writes a thresholded
feature-claim evidence report.

The current scoring is RAVEL-shaped: target-prompt movement is the cause score,
matched non-negation control movement is the isolation score, and the historical
`specificity_gap` is treated as cause minus isolation. The custom evaluator is
kept only as a negation adapter until SAEBench/RAVEL can be repointed cleanly.
Activation-density-matched control feature sets are now available and are
required for `strong_candidate_evidence`.

This is token-contrast evaluation, not broad behavioral understanding.
Phase 3 records target-prompt and matched-control intervention telemetry
separately, then writes aggregate row-level telemetry for summary thresholds.
It requires the three Phase 3 task families (`sentiment_negation`,
`property_negation`, and `state_negation`) to meet the configured minimum valid
task count. Expected model/SAE resource failures, task-validation failures,
compatibility failures, and non-finite baseline scores write artifact-backed
blocked runs instead of raw partial failures or fabricated rows.

## Semantic SAE Compatibility

Shape compatibility is necessary but not sufficient. Production decoded SAE
intervention requires:

- matching SAE-declared model identity,
- matching SAE-declared hook identity,
- matching hook layer and hook type when metadata is available,
- activation shape and encode/decode patch compatibility,
- finite reconstruction metrics.

`EleutherAI/pythia-70m` and `EleutherAI/pythia-70m-deduped` are different
checkpoints. The known compatible SAE below declares `pythia-70m-deduped`, so
all SAE commands for that release use `EleutherAI/pythia-70m-deduped`.

## Setup And Fast Tests

```bash
uv sync
uv run pytest
```

## Phase 1 Commands

```bash
uv run python scripts/check_real_model.py --device cpu

uv run python scripts/run_real_activation_ranking.py \
  --device cpu \
  --out runs/real_activation_ranking_pythia70m

uv run python scripts/run_real_residual_intervention.py \
  --ranking-dir runs/real_activation_ranking_pythia70m \
  --device cpu \
  --out runs/real_residual_intervention_pythia70m
```

The residual command is a smoke diagnostic even though it patches real
TransformerLens activations.

The preferred diagnostic alias is:

```bash
uv run python scripts/diagnostics/run_residual_smoke_patch.py \
  --ranking-dir runs/real_activation_ranking_pythia70m \
  --device cpu \
  --out runs/residual_smoke_patch_pythia70m
```

## Phase 2 SAE Compatibility

The repo has one verified small SAE path:

- model: `EleutherAI/pythia-70m-deduped`
- hook point: `blocks.2.hook_resid_post`
- SAE release: `pythia-70m-deduped-res-sm`
- SAE id: `blocks.2.hook_resid_post`

The Phase 2 evidence run is documented in `docs/phase2_run_evidence.md`.

```bash
uv run python scripts/check_sae_compatibility.py \
  --model EleutherAI/pythia-70m-deduped \
  --hook-point blocks.2.hook_resid_post \
  --sae-release pythia-70m-deduped-res-sm \
  --sae-id blocks.2.hook_resid_post \
  --device cpu \
  --out runs/check_sae_compatibility.json
```

Equivalent CLI:

```bash
uv run self-ground check-sae-compatibility \
  --model EleutherAI/pythia-70m-deduped \
  --hook-point blocks.2.hook_resid_post \
  --sae-release pythia-70m-deduped-res-sm \
  --sae-id blocks.2.hook_resid_post \
  --device cpu \
  --out runs/check_sae_compatibility.json
```

## Phase 2 SAE Ranking

```bash
uv run python scripts/run_real_activation_ranking.py \
  --model EleutherAI/pythia-70m-deduped \
  --hook-point blocks.2.hook_resid_post \
  --device cpu \
  --feature-source sae \
  --sae-release pythia-70m-deduped-res-sm \
  --sae-id blocks.2.hook_resid_post \
  --out runs/real_sae_ranking_pythia70m
```

## Phase 2 SAE Intervention

```bash
uv run python scripts/run_real_sae_intervention.py \
  --ranking-dir runs/real_sae_ranking_pythia70m \
  --out runs/real_sae_intervention_pythia70m \
  --model EleutherAI/pythia-70m-deduped \
  --hook-point blocks.2.hook_resid_post \
  --sae-release pythia-70m-deduped-res-sm \
  --sae-id blocks.2.hook_resid_post \
  --top-k-features 5 \
  --operation ablate \
  --patch-mode delta \
  --device cpu
```

## Phase 3 Token-Contrast Evaluation

Run this after producing the SAE ranking above:

```bash
uv run python scripts/run_phase3_behavioral_evaluation.py \
  --ranking-dir runs/real_sae_ranking_pythia70m \
  --out runs/phase3_behavioral_evaluation_pythia70m \
  --model EleutherAI/pythia-70m-deduped \
  --hook-point blocks.2.hook_resid_post \
  --sae-release pythia-70m-deduped-res-sm \
  --sae-id blocks.2.hook_resid_post \
  --per-family 2 \
  --top-k-features 2 \
  --baseline-mode top-vs-density-matched-multiseed \
  --random-seeds 7,11,13 \
  --density-tolerance 0.10 \
  --abs-mean-tolerance 0.10 \
  --operations ablate \
  --patch-mode delta \
  --device cpu \
  --write-report
```

RAVEL-shaped wrapper:

```bash
uv run python scripts/run_negation_ravel_eval.py \
  --ranking-dir runs/real_sae_ranking_pythia70m \
  --out runs/negation_ravel_eval \
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

SAEBench/RAVEL feasibility probe:

```bash
uv run python scripts/probe_saebench_ravel_bridge.py \
  --out runs/probe_saebench_ravel_bridge
```

The first serious GPU run plan is specified in
`experiments/E002_real_negation_sae_density_matched_run.md`. CPU runs with
`per_family=2` and `top_k_features=2` are diagnostic only.

The calibrated serious follow-up is specified in
`experiments/E003_calibrated_negation_sae_run.md`:

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

Latest E003 result:

- task bank calibration passed with 69 kept tasks:
  `property_negation=10`, `sentiment_negation=36`, `state_negation=23`;
- baseline intended-direction pass rate in the evaluation was `1.0`;
- run classification: `serious_gpu_evidence_run`;
- claim status: `insufficient_evidence`;
- top target delta: `0.6277369900026183`;
- top matched-control delta: `0.7188387469968934`;
- specificity gap: `-0.09110175699427508`.

Interpretation: E003 repaired the broken baseline task suite, including
`property_negation`, but did not produce negation-specific evidence because
matched-control movement remained larger than target-prompt movement.

Inspect a completed Phase 3 claim run without recomputing model results:

```bash
uv run python scripts/inspect_claim_run.py \
  --run-dir runs/diagnostic_negation_ravel_eval_density_matched
```

Equivalent CLI:

```bash
uv run self-ground run-phase3-behavioral-evaluation \
  --ranking-dir runs/real_sae_ranking_pythia70m \
  --out runs/phase3_behavioral_evaluation_pythia70m \
  --model EleutherAI/pythia-70m-deduped \
  --hook-point blocks.2.hook_resid_post \
  --sae-release pythia-70m-deduped-res-sm \
  --sae-id blocks.2.hook_resid_post \
  --per-family 2 \
  --top-k-features 2 \
  --baseline-mode top-vs-random-multiseed \
  --random-seeds 7,11,13 \
  --operations ablate \
  --patch-mode delta \
  --device cpu \
  --write-report
```

## Optional Integration Tests

Real SAE integration tests require:

```bash
export SELF_GROUND_SAE_MODEL=EleutherAI/pythia-70m-deduped
export SELF_GROUND_SAE_RELEASE=pythia-70m-deduped-res-sm
export SELF_GROUND_SAE_ID=blocks.2.hook_resid_post
uv run pytest --run-integration
```

Without those variables, SAE integration tests skip and the Phase 2 blocker workflow remains the executable path.

## Artifacts

Phase 1 model check:

- `runs/check_real_model.json`

Phase 1 residual ranking:

- `config.json`
- `pairs.jsonl`
- `activation_metadata.json`
- `feature_rankings.csv`
- `top_examples.jsonl`
- `README.md`

Phase 1 residual smoke diagnostic:

- `config.json`
- `selected_features.json`
- `intervention_results.jsonl`
- `summary.csv`
- `README.md`

Phase 2 compatibility:

- `runs/check_sae_compatibility.json`

Phase 2 SAE intervention:

- `config.json`
- `compatibility.json`
- `selected_features.json`
- `intervention_results.jsonl`
- `summary.csv`
- `README.md`

If compatibility fails, Phase 2 writes only `config.json`, `compatibility.json`, and `README.md`.

Phase 3 token-contrast evaluation:

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
- `control_suite.json`
- `control_task_mapping.jsonl`
- `control_suite_validation.json`
- `selected_feature_rationale.csv`
- `blocker.json` for blocked runs
- `mechanism_report.json`
- `mechanism_report.md`
- `README.md`

Successful Phase 3 reports require the full artifact contract above, except
`blocker.json`. The report builder blocks candidate claims if any required
artifact is missing.

## Interpretation Boundaries

Feature-space proxy arithmetic is not causal evidence.

Residual-dimension smoke patching is a real TransformerLens diagnostic, but it
is not SAE feature intervention, not candidate evidence, and not mechanism
discovery.

Decoded SAE intervention is real sparse-feature intervention only when compatibility succeeds and the run writes decoded SAE intervention artifacts.

Blocked compatibility artifacts are expected safety behavior when metadata,
shape, or reconstruction checks fail.

Phase 3 reports are thresholded and cautious. A diagnostic metadata mismatch
run, a tiny smoke run, failed task validation, zero/non-finite deltas, or weak
top-vs-control comparison cannot support strong candidate evidence.
Strong candidate evidence also requires at least three activation-density-matched
control feature sets. Without them, candidate evidence remains possible
but carries an explicit limitation that top-vs-random comparisons may be
confounded by baseline feature activity.
Rows with non-finite telemetry or logits are explicitly counted in
`skipped_behavioral_rows.json`; all-skipped runs are blocked and partially
skipped runs cannot support strong candidate evidence.
Baseline scores are stricter: any non-finite baseline score writes
`baseline_validation.json`, creates a blocked report, and skips decoded
intervention rows. `mechanism_report.md` includes configuration, semantic SAE
compatibility, reconstruction metrics, task validation, baseline calibration,
feature sets, target/control evidence, feature-set comparisons, telemetry,
threshold checks, limitations, unsupported claims, row accounting, and rerun
commands.

## E004 Specificity Diagnosis

E003 repaired the task-suite calibration problem but remained
`insufficient_evidence` because matched controls moved more than target prompts.
E004 diagnoses whether specificity can be rescued by stricter controls,
pre-intervention specificity-ranked feature sets, nearby residual-stream SAE
layers, and ablation/amplification operations.

Run the bounded CUDA matrix:

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

Compare cells with:

```bash
uv run python scripts/compare_e004_matrix.py \
  --matrix-root runs/e004_specificity_rescue_matrix \
  --e003 runs/e003_negation_eval_pythia70m_l2_calibrated_pf10_top5_density \
  --out runs/e004_specificity_rescue_matrix/comparison
```

Framework-shaped schemas, generic backend/plugin abstractions, and generic
trackers are not active in this repo.

## What Would Justify Future Framework Extraction?

Framework extraction should be reconsidered only after:

- at least two or three distinct tasks have run end to end,
- common artifacts and pain points are visible in real runs,
- no existing library owns the needed abstraction,
- extraction deletes code or reduces complexity rather than adding interfaces.
