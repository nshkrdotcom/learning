# E001: Phase 3 Token-Contrast Mechanism Evidence Evaluation

## Status
Planned. (A full implementation spec already exists outside this file —
this document is that spec compressed into a one-page, falsifiable design,
written *before* any Phase 3 code exists in `src/`.) **Hard prerequisite:**
`experiments/E000_baseline_behavioral_calibration.md` must run first — see
decision D007. Do not start building `baselines.py` / telemetry / report
code before E000's result is in `logs/claim_ledger.md`.

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
Deterministic generated token-contrast tasks across ≥3 negation-sensitive
families: `sentiment_negation`, `property_negation`, `state_negation` —
plus one trivially-easy, non-negation positive-control family (decision
D008, e.g. a strongly-primed factual completion) that small LMs are known
to handle reliably. The positive-control family is reported separately,
never pooled into the negation-family aggregates — it exists to make a
negation null result legible as "negation isn't represented this way," not
"the harness can't detect anything." Each task has a target prompt
(single-token target vs. foil under the negation condition) and a matched
non-negation control prompt (same surface structure, no negation). No
free-form grading, no LLM-as-judge.

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
- `random_seed_{1..N}`, N≈30–50 (seeded, excluding top fraction, same
  ranking artifact) — large enough to form a real empirical null
  distribution, not 3 fixed seeds (see decision D009). Report the top
  set's score as a **percentile rank** within this null distribution, not
  a point-estimate ratio.
- `bottom_active` (lowest nonzero-activation features from the same
  ranking)
- A paired statistic (sign test or Wilcoxon signed-rank) on per-task
  `specificity_gap` between `top` and each control set (decision D009a) —
  required in addition to the percentile rank, not instead of it.

Minimum for any claim: top vs. the full empirical null. Minimum for a
strong claim: percentile rank ≥ the strong threshold *and* the paired test
significant at the configured level, replicated across ≥3 task families.

## Controls
- Matched non-negation control prompt per task (same surface structure,
  negation condition removed) — this is the primary control for Phase 3
  (see D005). Unrelated collateral-drift probes are explicitly out of scope
  for this experiment.
- Token validation: every target/foil/control-target/control-foil string
  must resolve to exactly one model token; multi-token strings are excluded
  and logged, not silently dropped.

## Success criterion
`candidate_evidence`, per the existing thresholded report design, revised
per D009/D009a: E000 (Claim 0.1/0.2) has run and is recorded; compatibility
passes without a metadata-mismatch override; task validation passes with
≥2 valid tasks in ≥2 negation families plus the positive-control family;
top feature set's percentile rank in the empirical null (N≥30 seeds) clears
the candidate threshold *and* the paired statistic is significant; collateral
ratio stays under the candidate threshold; baseline task calibration
(from E000) is disclosed alongside the result regardless of its value.

## Failure criterion
Any of: E000 was skipped, compatibility fails, task validation fails (fewer
than 2 valid tasks in a required family), all target deltas are
zero/NaN/Inf, top feature set's percentile rank does not clear the
candidate threshold, the paired statistic is not significant, collateral
ratio exceeds threshold, or only a tiny smoke-scale run (`per_family` 1–2)
was performed — in which case the report is capped at
`insufficient_evidence` and must not be summarized as more than that
anywhere in the paper draft.

## Notes
- Implementation order matters: build and inspect `behavioral_tasks.py` +
  `task_validation.py` first, run E000, and manually check that default
  target/foil token strings actually tokenize as single tokens under the
  real Pythia tokenizer before wiring up the intervention loop. Don't
  assume the illustrative token lists in the spec survive contact with the
  real tokenizer — adjust and re-validate.
- Do not reuse the Phase 2 hand-picked pair (`sae_12300`, `sae_25521`) as
  the "top" set by default — recompute top features fresh from a
  `per_family` ranking large enough to not just reproduce the smoke-test
  selection, to avoid anchoring the new evaluation on a result that already
  has a paper-trail expectation attached to it. (Separately, look these two
  features up on Neuronpedia per decision D010 — that's free information
  regardless of what Phase 3 does.)
- A `strong_candidate_evidence` verdict requires ≥30 valid tasks, ≥3
  families, a full empirical null (≥30 seeds), both operations tested, and
  no metadata-mismatch override (D006). Budget the first real run as a
  verification pass at `per_family=2`–`5`, expect `insufficient_evidence` or
  `blocked`, and only scale up once the pipeline itself is confirmed
  correct.
- This experiment answers "is the selected feature set special compared to
  other features?" It does not answer "does a plausible explanation of the
  feature predict the right causal direction?" — that's E002, run as a
  separate, parallel track once this experiment's ranking artifact exists.
