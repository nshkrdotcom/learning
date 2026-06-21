# Phase 3: Token-Contrast Evaluation

## Current Phase Map

Current repo numbering is:

- Phase 1: real residual activation ranking and residual intervention.
- Phase 2: real decoded SAE feature intervention.
- Phase 3: multi-task token-contrast evaluation and candidate evidence reports.

Older notes may use different numbering. This document describes the current
Phase 3 implementation.

## What Phase 3 Adds

Phase 3 builds a reporting layer above Phase 2 decoded SAE intervention. It
does not introduce a new intervention primitive. It reuses the real path:

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

Use the term token-contrast evaluation. This is not broad behavioral
understanding and not complete mechanism discovery.

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
- optional bottom-active controls.

Random controls are deterministic by seed and never come from another SAE or
ranking file. Activation-matched controls are not implemented in Phase 3 and
remain a limitation.

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
  --baseline-mode top-vs-random-multiseed \
  --random-seeds 7,11,13 \
  --operations ablate \
  --patch-mode delta \
  --device cpu \
  --write-report
```

The script and Typer CLI also expose:

```bash
--max-relative-norm-drift-warning 0.5
--max-decoded-delta-norm-ratio-warning 0.5
```

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
- Feature sets are not activation-matched in Phase 3.
- Candidate reports are thresholded evidence summaries, not mechanism discovery.
