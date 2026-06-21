# Phase 3: Token-Contrast Evaluation

## Current Phase Map

Current repo numbering is:

- Phase 1: real residual activation ranking and residual smoke diagnostics.
- Phase 2: real decoded SAE feature intervention.
- Phase 3: multi-task token-contrast evaluation and candidate evidence reports.

Older notes may use different numbering. This document describes the current
Phase 3 implementation.

## What Phase 3 Adds

Phase 3 builds a reporting layer above Phase 2 decoded SAE intervention. It
does not introduce a new intervention primitive or a SELF-GROUND-owned engine.
It reuses the real path:

```text
real model activations
  -> real SAELens encode
  -> modify selected SAE features
  -> decode back to residual space
  -> patch the real TransformerLens model
  -> rerun logits
  -> score next-token contrasts
```

The evaluation asks whether selected SAE features produce more movement on
negation-sensitive target prompts than on matched non-negation control prompts,
and whether top-ranked features outperform deterministic control feature sets.
This is RAVEL-shaped: target-prompt movement is treated as cause score,
matched-control movement is treated as isolation score, and `specificity_gap`
is the cause-minus-isolation alias.

Use the term token-contrast evaluation. This is not broad behavioral
understanding and not complete mechanism discovery.

## Engine Boundary

The local backend is TransformerLens. SAE loading, encode, and decode are
SAELens responsibilities. SELF-GROUND owns the negation task schema,
compatibility gate, evaluation adapter, and claim ledger. It does not maintain a
generic patching or intervention engine. Residual-dimension outputs are smoke
diagnostics only, and feature-space proxy outputs are legacy-only.

Every Phase 3 report records `engine_backend` and `sae_backend`. A run using a
forbidden SELF-GROUND generic engine backend is blocked even if metric rows
look candidate-like.

## RAVEL-Shaped Wrapper

The forward-facing script is:

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
  --operations ablate \
  --patch-mode delta \
  --device cpu
```

This currently reuses the existing TransformerLens plus SAELens decoded
intervention path and records the SELF-GROUND negation RAVEL-style adapter in
`config.json`. A direct SAEBench bridge remains future work and must be tried
before expanding the custom evaluator.

## Compatibility Requirement

Production Phase 3 runs require semantic, shape, and reconstruction SAE
compatibility:

- requested model identity matches SAE metadata,
- requested hook point matches SAE metadata,
- hook layer and hook type match when available,
- activation width and encode/decode shapes are patch-compatible,
- reconstruction metrics are finite.

The known compatible small run uses:

```text
model: EleutherAI/pythia-70m-deduped
hook point: blocks.2.hook_resid_post
SAE release: pythia-70m-deduped-res-sm
SAE id: blocks.2.hook_resid_post
```

`EleutherAI/pythia-70m` and `EleutherAI/pythia-70m-deduped` are different
checkpoints.

`--allow-metadata-mismatch` is diagnostic-only. It may run to inspect shapes and
task behavior, but it records `diagnostic_only=true` and cannot support
`candidate_evidence` or `strong_candidate_evidence`.

## Tasks And Controls

Phase 3 generates deterministic token-contrast tasks for:

- `sentiment_negation`
- `property_negation`
- `state_negation`

Each task has:

- a negation target prompt,
- target and foil token strings,
- a matched non-negation control prompt,
- control target and foil token strings.

All token strings are resolved through the model tokenizer. Each scoring token
must map to exactly one model token. Excluded tasks are written to
`excluded_behavioral_tasks.jsonl`; they are not silently dropped.

Validation requires every Phase 3 family to meet
`min_valid_tasks_per_family`:

- `sentiment_negation`
- `property_negation`
- `state_negation`

A custom task file that omits or underfills any required family is blocked even
if all families present in the file tokenize cleanly. Validation summaries
include required families with zero counts when they are absent.

## Baselines And Feature Sets

Feature sets are selected only from the provided SAE ranking artifact:

- `top`
- seeded random controls excluding the top ranking fraction,
- activation-density-matched controls,
- optional bottom-active controls.

Random controls are deterministic by seed and never come from another SAE or
ranking file. Activation-density-matched controls are also deterministic by
seed. They match the top feature set on activation absolute mean and nonzero
fraction using the same ranking/task activation distribution.

Current ranking artifacts contain per-condition means rather than true
per-example activation densities. When true per-example density fields are not
available, SELF-GROUND records `stats_source=per_condition_mean_approximation`
in `feature_sets.json`. This prevents treating approximate density matching as
exact density matching.

Strong candidate evidence requires at least three actual
`density_matched_seed_*` control result rows. Candidate evidence can still be
reported without these controls, but the report includes this limitation:

```text
Activation-density-matched control feature sets are absent. Top-vs-random
comparisons may be confounded by baseline feature activity.
```

Baseline scoring is fail-closed. If any baseline target score, foil score,
prompt contrast, control target score, control foil score, or control contrast
is NaN or Inf, the run writes `baseline_validation.json`, writes a blocker
README/report, and does not run decoded interventions. This avoids treating
post-hoc skipped intervention rows as evidence when calibration itself failed.

## Telemetry

Each intervention row records separate target-prompt and matched-control
telemetry in:

- `target_intervention_telemetry`
- `control_intervention_telemetry`

The row also records `telemetry_provenance` and aggregate perturbation
telemetry derived from the two separate interventions:

- selected feature activation mean and absolute mean,
- selected feature modified mean and delta absolute mean,
- decoded delta norm,
- original and patched activation norms,
- relative norm drift.

Large norm drift is reported as a limitation and blocks strong evidence.

Rows with non-finite telemetry, logits, or derived scores are not silently
dropped. They are counted in `skipped_behavioral_rows.json`, summarized in the
run README, and included in `mechanism_report.json` row accounting. An
all-skipped run is `blocked`; a partially skipped run cannot become
`strong_candidate_evidence`.

## Artifact Layout

Successful run:

```text
config.json
behavioral_tasks.jsonl
behavioral_task_validation.json
excluded_behavioral_tasks.jsonl
compatibility.json
feature_sets.json
baseline_task_scores.jsonl
baseline_task_summary.csv
baseline_validation.json
behavioral_intervention_results.jsonl
behavioral_summary.csv
skipped_behavioral_rows.json
mechanism_report.json
mechanism_report.md
README.md
```

Blocked runs still write executable diagnostics, but do not write fabricated
intervention rows. Expected operational blockers such as model-load failures,
SAE-load failures, task-validation failures, compatibility failures, non-finite
baseline scores, and decoded-intervention resource failures write
`blocker.json`, `README.md`, and, when requested, `mechanism_report.json` and
`mechanism_report.md`.

## Report Status

`mechanism_report.json` uses thresholded claim statuses:

- `blocked`
- `insufficient_evidence`
- `candidate_evidence`
- `strong_candidate_evidence`

A tiny smoke run cannot produce `strong_candidate_evidence`. Diagnostic metadata
mismatch runs cannot produce candidate evidence. Reports include unsupported
claims explicitly, including no complete mechanism discovery, no broad model
understanding, no genuine introspection, and no monosemanticity claim.

Strong evidence also requires actual artifact rows, not config-only claims:

- finite aggregate summary rows,
- top-feature rows for both ablation and amplification,
- at least three actual random control feature-set result labels,
- norm drift below the configured strong-evidence threshold,
- zero norm-drift warning rate,
- no skipped rows,
- sufficient task and family counts.

Missing required artifacts produce `blocked`. Malformed or non-finite summary
rows produce `blocked` or `insufficient_evidence` rather than candidate status.
The required artifact gate includes `behavioral_tasks.jsonl`,
`excluded_behavioral_tasks.jsonl`, `baseline_task_summary.csv`, and
`skipped_behavioral_rows.json`; candidate claims cannot be made from partial
run directories.

`mechanism_report.md` includes explicit sections for configuration, semantic
SAE compatibility, reconstruction metrics, task validation, baseline
calibration, feature sets, target-prompt evidence, matched-control evidence,
feature-set comparison, intervention telemetry, threshold checks, claim status,
limitations, unsupported claims, row accounting, and rerun commands.

## Commands

First produce a compatible SAE ranking:

```bash
uv run python scripts/run_real_activation_ranking.py \
  --model EleutherAI/pythia-70m-deduped \
  --hook-point blocks.2.hook_resid_post \
  --feature-source sae \
  --sae-release pythia-70m-deduped-res-sm \
  --sae-id blocks.2.hook_resid_post \
  --device cpu \
  --per-family 1 \
  --top-k-features 5 \
  --out runs/test_real_sae_ranking
```

Then run Phase 3:

```bash
uv run python scripts/run_phase3_behavioral_evaluation.py \
  --ranking-dir runs/test_real_sae_ranking \
  --out runs/test_phase3_behavioral_evaluation \
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

The script and Typer CLI also expose:

```bash
--max-relative-norm-drift-warning 0.5
--max-decoded-delta-norm-ratio-warning 0.5
--density-tolerance 0.10
--abs-mean-tolerance 0.10
--allow-relaxed-density-matching / --no-allow-relaxed-density-matching
```

## SAEBench/RAVEL Probe

Before expanding the custom evaluator, run the bounded upstream feasibility
probe:

```bash
uv run python scripts/probe_saebench_ravel_bridge.py \
  --out runs/probe_saebench_ravel_bridge
```

The probe does not add SAEBench to core dependencies. It records whether
upstream packages are missing, importable but API-incompatible, or feasible for
a thin bridge that accepts custom negation tasks, a custom SAE, precomputed
activations, and cause/isolation scoring.

Optional integration:

```bash
SELF_GROUND_SAE_MODEL=EleutherAI/pythia-70m-deduped \
SELF_GROUND_SAE_RELEASE=pythia-70m-deduped-res-sm \
SELF_GROUND_SAE_ID=blocks.2.hook_resid_post \
uv run pytest --run-integration
```

## Limitations

- Next-token contrasts are narrow tests, not broad behavioral understanding.
- Task tokenization can exclude tasks for a given tokenizer.
- Baseline task calibration is recorded but does not prove model understanding.
- Density matching currently uses per-condition mean approximations unless
  future ranking artifacts include true per-example activation density fields.
- Candidate reports are thresholded evidence summaries, not mechanism discovery.
