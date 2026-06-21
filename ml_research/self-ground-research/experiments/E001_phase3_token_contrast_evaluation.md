# E001: Phase 3 Token-Contrast Mechanism Evidence Evaluation

## Status
Planned. (A full implementation spec already exists outside this file —
this document is that spec compressed into a one-page, falsifiable design,
written *before* any Phase 3 code exists in `src/`.)

## Research question
Do decoded SAE feature interventions on top-ranked negation-contrastive
features produce stronger, more specific next-token logit-contrast effects
than deterministic control feature sets, across multiple negation-sensitive
task families, while moving matched non-negation control prompts less?

## Hypothesis
The top-`k` SAE features selected by the existing negation-contrast ranking
score will show:
1. larger absolute target-prompt deltas than seeded-random and bottom-active
   control feature sets drawn from the *same* ranking artifact, and
2. a positive specificity gap (target delta minus matched-control delta)
   that is not attributable to generic intervention noise (i.e. survives
   telemetry/norm-drift checks).

## Model / SAE
- Model: `EleutherAI/pythia-70m-deduped`
- Hook point: `blocks.2.hook_resid_post`
- SAE release/id: `pythia-70m-deduped-res-sm` / `blocks.2.hook_resid_post`
  (the one verified-compatible pair on record — see `claim_ledger.md`
  Claim 2.1)

## Task
Deterministic generated token-contrast tasks across ≥3 families:
`sentiment_negation`, `property_negation`, `state_negation`. Each task has a
target prompt (single-token target vs. foil under the negation condition)
and a matched non-negation control prompt (same surface structure, no
negation). No free-form grading, no LLM-as-judge.

## Mechanism objects
SAE features (`sae_N` IDs) selected from a real SAE feature ranking on the
verified-compatible SAE above. Residual-dimension (`resid_N`) rankings are
explicitly rejected as input to this experiment — they are a different
mechanism object and Phase 1 already covers them.

## Claim format
Per result row: feature set, operation (`ablate`/`amplify`), patch mode
(`replace`/`delta`), target-prompt baseline/patched/delta scores, matched
control baseline/patched/delta scores, specificity gap, collateral ratio,
intervention telemetry (norm drift). No claim is emitted as prose — only as
these structured fields, aggregated into a thresholded report.

## Intervention
Real decoded SAE intervention via the existing Phase 2 primitives
(`sae_interventions.py`, `real_sae_intervention.py`): encode the real
activation, ablate/amplify selected feature indices, decode to residual
space, patch the real model with `patch_mode="delta"` by default (per D004),
rerun, score logits.

## Metrics
- `target_signed_delta`, `target_absolute_delta`
- `control_signed_delta`, `control_absolute_delta`
- `specificity_gap = target_absolute_delta - control_absolute_delta`
- `collateral_ratio = control_absolute_delta / target_absolute_delta`
  (null if target delta is 0 — must stay visible, not dropped)
- Intervention telemetry: relative norm drift, decoded delta norm — these
  gate report status, they don't just get reported as an FYI.

## Baselines
- `top` (ranking_abs_score_top_k)
- `random_seed_{7,11,13}` (seeded, excluding top fraction, same ranking
  artifact)
- `bottom_active` (lowest nonzero-activation features from the same
  ranking)

Minimum for any claim: top vs. at least one of these. Minimum for a strong
claim: top vs. ≥3 random seeds, per the thresholds below.

## Controls
- Matched non-negation control prompt per task (same surface structure,
  negation condition removed) — this is the primary control for Phase 3
  (see D005). Unrelated collateral-drift probes are explicitly out of scope
  for this experiment.
- Token validation: every target/foil/control-target/control-foil string
  must resolve to exactly one model token; multi-token strings are excluded
  and logged, not silently dropped.

## Success criterion
`candidate_evidence`, per the existing thresholded report design:
compatibility passes without a metadata-mismatch override, task validation
passes with ≥2 valid tasks in ≥2 families, top feature set beats at least
one control feature set by the configured ratio, collateral ratio stays
under the candidate threshold, baseline task calibration clears the minimum
intended-direction pass rate.

## Failure criterion
Any of: compatibility fails, task validation fails (fewer than 2 valid
tasks in a required family), all target deltas are zero/NaN/Inf, top
feature set does not beat any control set, collateral ratio exceeds
threshold, or only a tiny smoke-scale run (`per_family` 1–2) was performed
— in which case the report is capped at `insufficient_evidence` and must
not be summarized as more than that anywhere in the paper draft.

## Notes
- Implementation order matters: build and inspect `behavioral_tasks.py` +
  `task_validation.py` first, and manually check that default target/foil
  token strings actually tokenize as single tokens under the real Pythia
  tokenizer before wiring up the intervention loop. Don't assume the
  illustrative token lists in the spec survive contact with the real
  tokenizer — adjust and re-validate.
- Do not reuse the Phase 2 hand-picked pair (`sae_12300`, `sae_25521`) as
  the "top" set by default — recompute top features fresh from a
  `per_family` ranking large enough to not just reproduce the smoke-test
  selection, to avoid anchoring the new evaluation on a result that already
  has a paper-trail expectation attached to it.
- A `strong_candidate_evidence` verdict requires ≥30 valid tasks, ≥3
  families, ≥3 random seeds, both operations tested, and no metadata-mismatch
  override (D006). Budget the first real run as a verification pass at
  `per_family=2`–`5`, expect `insufficient_evidence` or `blocked`, and only
  scale up once the pipeline itself is confirmed correct.
