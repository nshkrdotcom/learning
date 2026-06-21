# Research Log

Short, structured, dated. The point is retrieval, not prose. Fill this in
during or right after every work session — not from memory a week later.

Format:

```text
## YYYY-MM-DD

### Question
### Context
### Hypothesis
### Work done
### Result
### Interpretation
### Decision
### Open questions
```

---

## (retrospective) Phase 1 close-out

### Question
Can the repo prove it touches a real transformer's real activations and
real logits, end to end, with no fake adapters anywhere in `src/`?

### Context
Early repo state had schemas, generators, and adapters but no human-runnable
script that produced a real artifact a reviewer could inspect without
reading test code.

### Hypothesis
A minimal real-model check script plus a real activation-ranking script plus
a real residual-intervention script would be enough to call Phase 1 closed.

### Work done
- Implemented `scripts/check_real_model.py`.
- Implemented `scripts/run_real_activation_ranking.py` with residual-dimension
  ranking by negation contrast.
- Implemented `scripts/run_real_residual_intervention.py` with real
  TransformerLens hook patching and logit-contrast measurement.
- Wrote `docs/milestone_v0.md`.

### Result
All three commands run and produce real artifacts
(`runs/check_real_model.json`, `runs/test_real_activation_ranking/`,
`runs/test_real_residual_intervention/`).

### Interpretation
Phase 1 is engineering-verified path validation. It says nothing yet about
which residual dimensions matter — see Claim 1.1–1.3 in `claim_ledger.md`.

### Decision
Move to Phase 2: real SAE feature ranking and decoded intervention, since
residual dimensions are not interpretable units on their own.

### Open questions
- Which pretrained SAE release/id is actually compatible with this model?

---

## (retrospective) Phase 2 close-out

### Question
Can the repo run a real decoded SAE feature intervention — encode, modify
selected features, decode, patch the real model, measure real logits — and
can it tell the difference between a genuinely compatible SAE and a
same-width but wrong-checkpoint SAE?

### Context
D001/D002 in `decision_log.md`. Needed a concrete, verified SAE release/id
before any decoded intervention could be claimed real rather than aspirational.

### Hypothesis
`pythia-70m-deduped-res-sm` / `blocks.2.hook_resid_post` would be compatible
with `EleutherAI/pythia-70m-deduped`, and the *non*-deduped checkpoint would
correctly fail the compatibility check despite matching tensor shapes.

### Work done
- Implemented `sae_compat.py` (semantic + shape + reconstruction checks).
- Implemented `sae_interventions.py` (encode/modify/decode primitives).
- Implemented `real_sae_intervention.py` and
  `scripts/run_real_sae_intervention.py`.
- Ran the intentional-mismatch check and the matched check.
- Ran SAE feature ranking and a 4-pair, 2-feature decoded ablation.
- Ran full test suite with `--run-integration` and SAE env vars set.

### Result
- Mismatch check failed closed as predicted (`compatible=false`, exit 1).
- Matched check passed (`compatible=true`, finite reconstruction metrics).
- SAE ranking produced a real top feature (`sae_12300`, score 0.232).
- Decoded ablation produced a real, nonzero, finite
  `signed_specificity_score_mean=0.0038` over 4 pairs.
- `80 passed` on the full suite with SAE env vars configured.

### Interpretation
The real decoded-SAE-intervention path works end to end on a verified
compatible SAE. This is `single_run_evidence` (Claim 2.3), not evidence that
`sae_12300`/`sae_25521` are negation-specific — there is no control feature
set in this run to compare against, and n=4 pairs is a smoke test, not a
sample.

### Decision
Do not write any prose claiming "the negation feature was found." Commit
the evidence as-is (commit `0cd9742`) and scope the next phase specifically
around the missing piece: baseline feature-set comparisons + multiple task
families + matched controls. That missing piece is Phase 3.

### Open questions
- What's the right control feature-set selection method (random? bottom-active
  by score? both)? -> answered in Phase 3 spec via `baselines.py` design.
- How many task families are enough to call something a candidate claim?
  -> answered via thresholds in the Phase 3 mechanism-report design (≥2 for
  candidate, ≥3 for strong).

---

## 2026-06-20

### Question
How do we keep the next phase of work (Phase 3) from drifting into
overclaiming, given that Phase 2 already produced one real, clean-looking
positive number (`signed_specificity_score_mean=0.0038`)?

### Context
The Phase 2 evidence is real but thin (n=4, no controls, two hand-picked
features). There's an existing, very detailed Phase 3 implementation spec
already written (task validation, baseline feature sets, intervention
telemetry, thresholded mechanism-evidence report with explicit
`blocked` / `insufficient_evidence` / `candidate_evidence` /
`strong_candidate_evidence` statuses) — but no code or runs against it yet.

### Hypothesis
Setting up the research record (claim ledger, run ledger, decision log,
prior-art matrix, paper draft, experiment registry) *before* writing Phase 3
code will make it harder to accidentally upgrade a claim past what the
evidence supports, because every claim has to be written down with its
exact supporting artifact before it can move.

### Work done
- Built `research/` scaffold: `README.md`, `logs/claim_ledger.md`,
  `logs/run_ledger.csv`, `logs/decision_log.md`, `logs/research_log.md`
  (this file), `literature/prior_art_matrix.md`, `paper/draft.md`,
  `paper/abstract_drafts.md`,
  `experiments/E001_phase3_token_contrast_evaluation.md`.
- Populated the claim ledger and run ledger from the real evidence already
  documented in `docs/phase2_run_evidence.md` and `docs/qc_checklist.md`,
  not from assumption.
- Compressed the existing Phase 3 implementation spec into the
  experiment-registry template (E001) so it states a falsifiable hypothesis,
  success/failure criteria, and exact thresholds in one page instead of a
  multi-thousand-word prompt.

### Result
A research record exists that currently shows: 3 engineering-verified
claims (Phase 1), 1 engineering-verified + 2 single-run-evidence claims
(Phase 2), and 2 explicitly unsupported claims (Phase 3) with a concrete
plan to test them.

### Interpretation
There is nothing to overclaim yet because nothing beyond
`single_run_evidence` exists. The paper draft (`paper/draft.md`) is written
to match that — it is a method/harness paper with a small Phase 2 proof of
the technical path, not a results paper.

### Decision
Next concrete action is to implement Phase 3 starting with
`behavioral_tasks.py` + `task_validation.py` (deterministic task generation
and tokenization validation) *before* touching the intervention/report code,
so that the task set itself can be inspected and fixed before it's used to
generate any claims. Do not run the full Phase 3 evaluation until the task
validation step has been verified against the real model tokenizer and the
default token lists have been adjusted so ≥2 tasks per family survive
validation.

### Open questions
- Will the default target/foil token strings in the Phase 3 spec actually
  be single tokens under the Pythia tokenizer, or will most need adjusting?
  (Likely yes for some — needs to be checked empirically, not assumed.)
- Should the first Phase 3 run reuse `sae_12300`/`sae_25521` as the "top"
  set (since they already have Phase 2 evidence), or recompute top features
  fresh from a larger `per_family` ranking? Leaning toward recomputing fresh
  to avoid anchoring the new evaluation on a hand-picked pair.
