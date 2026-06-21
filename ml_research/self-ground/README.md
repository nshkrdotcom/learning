# SELF-GROUND

SELF-GROUND is a negation-scope interpretability experiment harness. The repo
now has a Phase 1 real residual-dimension pipeline, Phase 2 decoded SAE
intervention infrastructure, and Phase 3 token-contrast evaluation reports. It
does not claim complete SELF-GROUND, broad mechanism discovery, broad behavioral
understanding, or genuine model introspection.

## Phase 1 Recap

Phase 1 implements:

- deterministic negation minimal pairs,
- real TransformerLens activation capture,
- real residual-dimension activation ranking,
- real residual-dimension intervention through TransformerLens hooks,
- real logit-contrast deltas after residual patching.

Phase 1 does not claim sparse SAE mechanisms. Raw residual dimensions are basis-dependent.

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
mechanism evidence report.

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
  --baseline-mode top-vs-random-multiseed \
  --random-seeds 7,11,13 \
  --operations ablate \
  --patch-mode delta \
  --device cpu \
  --write-report
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

Phase 1 residual intervention:

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
- `blocker.json` for blocked runs
- `mechanism_report.json`
- `mechanism_report.md`
- `README.md`

Successful Phase 3 reports require the full artifact contract above, except
`blocker.json`. The report builder blocks candidate claims if any required
artifact is missing.

## Interpretation Boundaries

Feature-space proxy arithmetic is not causal evidence.

Residual-dimension intervention is real TransformerLens intervention evidence, but it is not SAE feature intervention and not mechanism discovery.

Decoded SAE intervention is real sparse-feature intervention only when compatibility succeeds and the run writes decoded SAE intervention artifacts.

Blocked compatibility artifacts are expected safety behavior when metadata,
shape, or reconstruction checks fail.

Phase 3 reports are thresholded and cautious. A diagnostic metadata mismatch
run, a tiny smoke run, failed task validation, zero/non-finite deltas, or weak
top-vs-control comparison cannot support strong candidate evidence.
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
