# Claim Ledger

This is the most important file in the research record. Every claim that
could appear in the paper gets a row here, and the row can only be upgraded
when new evidence actually lands — not when it would be convenient for the
narrative.

## Status definitions

```text
unsupported            - no real evidence yet, or evidence contradicts it
engineering_verified   - the code path runs correctly on real data; not yet a scientific finding
single_run_evidence    - one real run produced a real (nonzero, finite) result; n is small, no baselines
candidate_claim        - passes Phase 3 thresholds: beats deterministic control feature sets,
                         multiple task families, matched controls, telemetry within bounds
replicated_evidence    - candidate_claim, replicated across seeds/runs/checkpoints
failed_or_weakened     - was a candidate claim, then a control/replication run weakened or killed it
```

A claim's status can only be set by someone who has looked at the artifact,
not by someone who remembers running something like that once.

---

## Phase 1 — real residual ranking and intervention

**Claim 1.1**: The repo can load a real small transformer (TransformerLens),
run real strings through it, and capture a real activation tensor at a named
hook point.
**Status:** `engineering_verified`
**Evidence:** `docs/milestone_v0.md`, `runs/check_real_model.json` (per
documented command `scripts/check_real_model.py --device cpu`).
**Caveat:** none — this is a path-validation claim, not a scientific one.

**Claim 1.2**: The repo can rank real residual-stream dimensions by a
deterministic negation-contrast score (`mean(x_pos)+mean(x_para) -
mean(x_neg)-mean(x_decoy)`) over real activations.
**Status:** `engineering_verified`
**Evidence:** `docs/milestone_v0.md`, `runs/test_real_activation_ranking/`.
**Caveat:** residual-dimension ranking is not feature-level interpretability;
it has never been described as such in the docs, and this ledger preserves
that boundary.

**Claim 1.3**: The repo can patch a real residual-stream activation at a
named hook point during a real forward pass and measure a real logit-contrast
change.
**Status:** `engineering_verified`
**Evidence:** `runs/test_real_residual_intervention/` per
`scripts/run_real_residual_intervention.py`.
**Caveat:** residual-dimension intervention, not SAE-feature intervention —
do not conflate the two in prose.

---

## Phase 2 — real decoded SAE feature intervention

**Claim 2.1**: The repo can verify real semantic + shape + reconstruction
compatibility between a chosen model/hook point and a chosen SAELens
release/id, and fails closed (rejects) when the SAE's declared checkpoint
does not match the requested model.
**Status:** `engineering_verified`
**Evidence:** `docs/phase2_run_evidence.md` — the intentional-mismatch check
(`EleutherAI/pythia-70m` against an SAE declaring `pythia-70m-deduped`)
returned `metadata_compatible=false`, `compatible=false`, exit code `1`. The
matched check (`pythia-70m-deduped` against the same SAE) returned
`compatible=true` with finite reconstruction metrics
(`reconstruction_mse=0.1076`, `reconstruction_l2_relative=0.0867`).
**Caveat:** verified for exactly one release/id pair
(`pythia-70m-deduped-res-sm` / `blocks.2.hook_resid_post`). Do not generalize
to "the compatibility checker works for arbitrary SAEs" — it has only been
exercised against one real positive and one real negative case.

**Claim 2.2**: The repo can rank real SAE features by the same
negation-contrast score used for residual dimensions, using a real,
compatible pretrained SAE.
**Status:** `single_run_evidence`
**Evidence:** `docs/phase2_run_evidence.md` — top-ranked feature
`sae_12300`, `score=0.2324` (`mean_pos=0.390`, `mean_para=0.205` vs.
`mean_neg=0.177`, `mean_decoy=0.186`).
**Caveat:** `per_family=1` — this is a smoke-scale ranking, not a claim about
which features are "the" negation features. No comparison against a control
feature set has been run yet (that's Phase 3, Claim 3.x below).

**Claim 2.3**: The repo can perform a full real decoded SAE intervention —
encode real activations, ablate selected SAE features, decode back to
residual space, patch the real model, rerun, and measure a real, nonzero
logit-contrast delta.
**Status:** `single_run_evidence`
**Evidence:** `docs/phase2_run_evidence.md` —
`runs/test_real_sae_intervention/summary.csv`:
`feature_set=sae_12300+sae_25521, operation=ablate, patch_mode=delta,
n_pairs=4, signed_specificity_score_mean=0.00379` (nonzero, finite).
**Caveat:** `n_pairs=4`, two hand-selected top features, ablate only, no
amplify, no random/bottom-active control feature set, single task family
(the original negation minimal pairs, not the Phase 3 token-contrast
tasks). This is proof the real pipeline executes end to end and produces a
real measurable effect — it is explicitly **not** evidence that the effect
is negation-specific or that it beats chance. Treat any temptation to call
this "the negation feature" as the failure mode this ledger exists to
prevent.

---

## Phase 3 — token-contrast evaluation (not yet run)

**Claim 3.1**: Top-ranked SAE features (selected by negation-contrast score)
produce larger target-prompt logit-contrast deltas than deterministic random
or bottom-active control feature sets, across multiple negation-sensitive
task families, while moving matched non-negation control prompts less.
**Status:** `unsupported` — this is the central hypothesis Phase 3 exists to
test. No run has produced evidence for or against it yet.
**Needed evidence:** see `experiments/E001_phase3_token_contrast_evaluation.md`
for the full design — in short: ≥2 task families with ≥2 valid tasks each,
a top-vs-random(-vs-bottom-active) feature-set comparison from the *same*
ranking artifact, matched non-negation controls scored alongside targets,
intervention telemetry within drift thresholds, and baseline task
calibration above the configured pass-rate.
**Promotion rule:** can become `candidate_claim` only under the thresholds
defined in the Phase 3 spec (`min_top_vs_control_ratio`,
`max_collateral_ratio_for_candidate`, etc.) — never by inspection or by a
single favorable run.

**Claim 3.2**: The above effect replicates with ≥30 valid tasks, ≥3 task
families, ≥3 random baseline seeds, and survives both ablation and
amplification.
**Status:** `unsupported`
**Note:** this is the bar for `strong_candidate_evidence` in the Phase 3
report schema. A `per_family=1` or `per_family=2` smoke run can never
satisfy this — don't let a convenient small run get summarized as if it
did.

---

## Out of scope for the first paper (Level 4 — do not pursue yet)

**Claim 4.1**: "This SAE feature set is *the* negation mechanism in this
model." / "The model understands negation." / "SELF-GROUND solves
self-interpretation."
**Status:** `unsupported`, and not currently being tested for. These claims
need cross-layer, cross-checkpoint, and likely cross-architecture evidence
that is well beyond a first paper. If Phase 3 produces
`strong_candidate_evidence`, the correct write-up is still "candidate
evidence consistent with a narrowly specified causal role," not mechanism
discovery.

---

## How to update this file

1. Run something real.
2. Open the artifact (not your memory of the run).
3. Find the matching claim row, or add a new one.
4. Move the status only as far as the evidence in hand justifies — one rung
   at a time.
5. If a run *weakens* a claim (e.g., a control beats the "top" feature set),
   write that down as `failed_or_weakened` with the same rigor you'd give a
   positive result. Negative results are real findings here, not failures
   to hide.
