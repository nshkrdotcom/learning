# Paper Draft (Living)

Status: hypothesis-management document, not a results draft. Sections
marked [PLACEHOLDER] are allowed to be empty or wrong; sections without that
marker should only ever say what `logs/claim_ledger.md` currently supports.

## Title candidates

1. SELF-GROUND: Grounding SAE Feature Interpretations with Controlled
   Decoded Interventions
2. A Reproducible Harness for Testing SAE Feature Claims Against Decoded
   Activation Interventions
3. Do Models Know Their Own Mechanisms? Evaluating SAE Feature
   Interpretations by Intervention Prediction

Working preference: (1) or (2). Title (3) oversells what this first paper
can support — see the claim ladder below — and should be retired unless a
later paper actually gets a model to *emit* the structured claims itself
rather than the researcher selecting features from a ranking.

## The thesis (current, narrow version)

> SAE feature interpretations should not be accepted from top-activating
> examples or feature labels alone. They should be evaluated through
> controlled matched examples, real decoded activation interventions,
> deterministic control feature sets, and matched non-negation controls.
> SELF-GROUND implements this evaluation loop for a small negation-scope
> case study and reports which claims are supported, weakened, or blocked
> by the evidence — using a fixed evidence ladder so that a small favorable
> run cannot be summarized as more than it is.

What this thesis is *not*: "models can understand themselves" (too broad,
unfalsifiable as stated) or "we found the negation mechanism" (too strong
for any evidence currently in hand, and probably too strong for what a
single small-model case study could ever establish).

## The claim ladder

Use this to keep every section honest. See `logs/claim_ledger.md` for the
exact evidence behind each level — this is the compressed version.

**Level 1 — supported.** A real pipeline captures real activations, ranks
real residual dimensions and real SAE features by a deterministic
negation-contrast score, and performs real TransformerLens interventions.

**Level 2 — supported (single-run).** With a verified-compatible pretrained
SAE, the pipeline encodes real activations, modifies selected SAE features,
decodes back to residual space, patches the real model, reruns, and
measures real nonzero logit-contrast deltas. Verified for one model/SAE
pair, n=4 pairs, two hand-picked features, ablation only.

**Level 3 — the target of the current experiment (Phase 3 / E001), not yet
supported.** Some top-ranked SAE feature sets produce stronger,
matched-control-specific token-contrast effects than deterministic control
feature sets, across multiple task families.

**Level 4 — explicitly out of scope for this paper.** "This is the model's
negation mechanism." Needs cross-layer, cross-checkpoint, likely
cross-architecture evidence well beyond what a first paper, or this case
study, can support.

This paper aims at Level 3, with the harness itself (Levels 1–2,
generalized) as the main contribution and the negation case study as the
demonstration, not the headline.

## Working abstract (current)

> Sparse autoencoders are increasingly used to assign semantic
> interpretations to internal features of language models, but feature
> descriptions based on top-activating examples alone provide limited
> evidence that a feature plays the claimed causal role. We introduce
> SELF-GROUND, a small reproducible evaluation harness for testing
> candidate SAE feature interpretations through controlled examples,
> decoded feature interventions, matched controls, and explicit evidence
> thresholds. As a case study, we evaluate negation-sensitive features in a
> small transformer language model. SELF-GROUND generates matched negation
> minimal pairs and multi-task token-contrast tasks, ranks residual
> dimensions and SAE features by controlled activation contrast, verifies
> SAE/model/hook semantic and reconstruction compatibility, modifies
> selected SAE activations, decodes them back to residual space, patches
> the model during a real forward pass, and measures next-token logit
> contrast against deterministic control feature sets and matched
> non-negation control prompts. This work does not claim to discover a
> complete negation mechanism, broad model understanding, or genuine model
> introspection. Instead, it proposes a disciplined evidence ladder — from
> activation specificity, to decoded intervention effect, to control-beating
> specificity under thresholded reporting — and an auditable scaffold for
> recording when the evidence for a feature claim is, or is not, sufficient.

See `paper/abstract_drafts.md` for an alternate, more MI-reviewer-facing
phrasing.

## Outline

```text
1. Introduction
   - Problem: SAE feature labels are easy to overinterpret from examples alone.
   - Need: controlled, causal, falsifiable evidence with explicit thresholds.
   - Case study: negation scope in a small transformer.
   - Contribution: reproducible harness + evidence ledger + decoded
     intervention pipeline + thresholded mechanism-evidence report.

2. Background
   - Residual streams and activation patching.
   - Sparse autoencoders; pretrained SAEs (SAELens) vs. training new ones.
   - Feature-interpretation risks: descriptive collision, polysemanticity,
     metadata/checkpoint mismatch, off-distribution decoded patches.
   - Why negation minimal pairs are a useful small first target.

3. SELF-GROUND Evidence Ladder
   - Controlled data (minimal pairs + token-contrast tasks).
   - Activation specificity (ranking score).
   - Semantic + shape + reconstruction SAE compatibility (fail-closed).
   - Decoded intervention (encode/modify/decode/patch/measure).
   - Deterministic control feature sets (random, bottom-active).
   - Matched non-negation controls.
   - Thresholded claim statuses: blocked / insufficient_evidence /
     candidate_evidence / strong_candidate_evidence.

4. Methods
   - Model and SAE (exact release/id; fail-closed metadata check).
   - Minimal-pair and token-contrast task generation; tokenization
     validation and exclusion bookkeeping.
   - Ranking score definition.
   - Decoded intervention procedure; replace vs. delta patch mode (D004).
   - Baselines: top / random-seeded / bottom-active feature sets.
   - Metrics: signed/absolute deltas, specificity gap, collateral ratio.
   - Intervention telemetry and norm-drift thresholds.
   - Claim-status promotion rules (exact thresholds, not vibes).

5. Implementation
   - TransformerLens + SAELens.
   - Artifact layout; fail-closed blockers instead of fabricated rows.
   - Reproducibility: exact commands, exact verified release/id.

6. Results [PLACEHOLDER — do not write until Phase 3 actually runs]
   - Phase 1/2 path-validation results (already real — can be written now,
     see logs/claim_ledger.md Claims 1.1-1.3, 2.1-2.3).
   - Phase 3 token-contrast results (placeholder table shells only).
   - Top-vs-random/bottom-active comparisons (placeholder).
   - Telemetry/norm-drift summary (placeholder).

7. Discussion [PLACEHOLDER]
   - What the evidence supports vs. does not.
   - Failure modes encountered (record honestly, including any
     insufficient_evidence or blocked outcomes).

8. Limitations
   - Single small model, single concept domain, single verified SAE.
   - Token-contrast tasks are narrow; not a broad behavioral claim.
   - SAE reconstruction is lossy (~8.7% relative L2 error on the verified
     SAE) — delta patch mode mitigates but does not eliminate this.
   - No claim of monosemantic feature discovery or cross-checkpoint
     generalization.

9. Conclusion
```

## Contribution statement (draft)

This paper makes three contributions, scoped to match the claim ladder
above. First, it formalizes evaluation of SAE feature interpretations as a
controlled, intervention-grounded evidence problem rather than an
example/label plausibility check. Second, it implements a fail-closed
pipeline — real model, real SAE, real decoded intervention, real
compatibility verification that rejects same-width wrong-checkpoint SAEs —
with artifacts a reviewer can rerun and inspect directly. Third, it defines
an explicit, mechanically-enforced evidence ladder (including a floor that
a small smoke run cannot cross) for reporting whether a candidate feature
set's effects are real, control-beating, and specific, or merely a
single-run curiosity.

## What not to write yet

- Results/Discussion prose beyond placeholders, until Phase 3 actually runs.
- Anything implying `sae_12300`/`sae_25521` are "the" negation features —
  current evidence is single-run, n=4, no controls (Claim 2.3).
- Anything using "mechanism," "understands," or "discovered" without the
  qualifying "candidate evidence consistent with..." framing.
