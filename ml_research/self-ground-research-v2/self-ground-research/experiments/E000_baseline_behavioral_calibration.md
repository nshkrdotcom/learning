# E000: Baseline Behavioral Calibration (prerequisite gate)

## Status
Planned. This must run — and be inspected by a human, not just thresholded
silently — before `baselines.py`, `intervention_telemetry.py`, or
`mechanism_report.py` are built for Phase 3. See decision D007.

## Research question
Does the unpatched model (`EleutherAI/pythia-70m-deduped`, no SAE, no
intervention) already show the intended-direction next-token preference on
the negation token-contrast tasks, and on a trivially-easy positive-control
task family?

## Hypothesis
Uncertain by design — that's the point of running this first. Two
informative outcomes:
1. High pass rate on negation tasks: proceed to Phase 3 feature work with
   confidence the behavioral target exists.
2. Low pass rate on negation tasks but high pass rate on the positive
   control: the harness works, but this model/prompt design doesn't reliably
   represent negation behaviorally — revise task templates (more context,
   different surface forms) before any SAE work, or treat the eventual
   Phase 3 result as testing a weaker behavioral signal and say so plainly
   in the paper.

A third, bad outcome (low pass rate on *both* negation and the positive
control) would mean a bug in scoring/tokenization, not a model finding —
fix the harness, don't write up a result.

## Model
`EleutherAI/pythia-70m-deduped`, no SAE required for this experiment at all
— this is pure forward-pass logit scoring, CPU, no hooks.

## Task
All `behavioral_tasks.py` families, including the new positive-control
family (decision D008), scored once each with zero patching.

## Mechanism objects
None — there is no intervention in this experiment. This is intentionally
the simplest possible run in the whole project.

## Claim format
Per task: `baseline_target_score`, `baseline_foil_score`,
`baseline_contrast`, `intended_direction_pass` (bool). Aggregated to
`intended_direction_pass_rate` per family.

## Intervention
None.

## Metrics
- `intended_direction_pass_rate` per family (negation families and the
  positive-control family reported separately, never pooled)
- Distribution of `baseline_contrast` per family (not just the mean — a
  family that's 50% strongly-correct and 50% strongly-wrong looks very
  different from one that's uniformly weakly-correct, and the mean alone
  hides that)

## Baselines
None — there is nothing to compare against in this experiment.

## Controls
The positive-control family itself is the control for the harness.

## Success criterion
Positive-control family clears a high pass-rate bar (e.g. ≥0.9) — this
confirms the harness/tokenizer/scoring path works. Negation families'
pass rate is recorded and reported honestly regardless of value; there is
no "success" threshold for negation here, only information.

## Failure criterion
Positive-control family fails to clear the high bar. This means stop and
debug the harness (tokenization, scoring, task templates) — do not proceed
to Phase 3 feature work, and do not write up any negation-related result
until this is fixed.

## Notes
- This experiment costs almost nothing (CPU, no SAE download, no hooks) and
  should be the very first thing run once `behavioral_tasks.py` and
  `task_validation.py` exist — before `baselines.py` is even started.
- Write the result into `logs/research_log.md` and update Claims 0.1/0.2 in
  `logs/claim_ledger.md` honestly, including if the negation pass rate is
  low. A low pass rate here, reported clearly, is itself a legitimate and
  citable finding about negation representation in small LMs — see the
  Kassner & Schütze line of work in `literature/prior_art_matrix.md`.
