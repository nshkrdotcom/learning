# Portfolio Write-Up Plan

This is a separate artifact from `paper/draft.md`. The academic-style draft
is for if/when this becomes an arXiv-style paper. This file is for the
thing that actually gets a solo mech interp project noticed: a clear,
honest, well-figured post in the format and venue this community actually
reads — LessWrong / the Alignment Forum, or a personal blog cross-posted
there, not a PDF nobody opens.

## Why this is a separate document

A repo with green tests is necessary but not sufficient for getting noticed.
The people worth getting noticed by (independent researchers, EleutherAI/
Neuronpedia/SAELens-adjacent folks, Anthropic/DeepMind/Redwood interp
people who read the Alignment Forum) engage with posts, not papers. The
norms are different: epistemic status up front, explicit "what would
change my mind," willingness to show a null result, and code linked
prominently — not abstract/intro/related-work/methods/results/discussion.

## Structure

```text
Title: <plain-language version of the finding, not a paper title>
       e.g. "Testing whether a sparse autoencoder feature for negation
       survives intervention" — not "SELF-GROUND: Grounding SAE..."

Epistemic status: <one line — e.g. "Fairly confident in the
  infrastructure and the calibration result; uncertain about the
  candidate-feature claim, see thresholds below.">

TL;DR: <3-5 sentences. State the actual claim-ladder status achieved,
  not the aspiration.>

## Why I did this
- The descriptive-collision problem (McCann 2026) in one paragraph.
- Why negation, why this model, why this is tractable solo.

## What I built
- One paragraph + the pipeline diagram (reuse/adapt the structural
  diagram already produced this session).
- Link the repo and the exact commit/tag up front, not at the bottom.

## The behavioral calibration result (E000) — show this even if it's bad
- Pass rate on negation families vs. the positive control, with a
  bar chart. This section exists and is shown regardless of outcome.

## The feature-set comparison (E001)
- Empirical null histogram with the top set's position marked.
- Per-family specificity-gap table.
- State the exact claim_status the report reached, in the report's own
  vocabulary (blocked / insufficient_evidence / candidate_evidence /
  strong_candidate_evidence) — do not round up in prose.

## The explainer-prediction result (E002), if run
- This is probably the most interesting section to this audience —
  lead figures with it if the numbers are at all interesting, including
  if they're a clean null.

## What I'd do differently / limitations
- Be specific and a little self-critical. This is the section that
  signals "this person can be trusted," more than any result above it.

## What would change my mind
- One paragraph: what evidence would make you believe the candidate
  feature claim more, or less.

## Code and reproduction
- Exact repo link, exact commit/tag, exact commands. Near the top of the
  post too, not just here — don't make people scroll to find it.
```

## Figures to produce once results exist

1. Pipeline diagram (have a version already from this planning session;
   regenerate with final real numbers once available).
2. Bar chart: intended-direction pass rate, negation families vs. positive
   control (from E000) — this is the single most important figure for
   credibility, and it goes early in the post even though it's "just" a
   calibration check.
3. Empirical null histogram with the top feature set's score marked
   (from E001/D009) — this is the figure that makes the statistics
   legible at a glance instead of requiring the reader to parse a table.
4. Per-family specificity-gap small-multiples (one mini bar chart per
   family) rather than one big aggregated number.
5. If pursued: layer-sweep line plot (effect size vs. layer, D012).
6. If E002 is run: scatter or simple table of predicted-sign vs.
   actual-sign per feature, ideally with the collision-candidate features
   marked distinctly.

## Reproducibility checklist for the post itself

- [ ] Tag the exact commit referenced in the post (e.g. `v0.3-phase3`),
      don't just say "main branch as of writing."
- [ ] Pin `sae-lens`/`transformer-lens` versions in the post, not just in
      `pyproject.toml` — readers copy commands from posts, not from CI.
- [ ] Link the exact SAE release/id and, if found, the Neuronpedia feature
      pages for any feature IDs discussed (decision D010).
- [ ] State the hardware/runtime (CPU is fine and worth saying explicitly —
      it's part of what makes this approachable for other solo
      researchers to replicate or extend).

## Where to post

LessWrong (cross-posts automatically to the Alignment Forum if it's
interpretability-tagged and meets their bar) is the default choice for this
community. Consider a short Twitter/X thread linking it once posted, but
the post itself is the artifact — don't lead with the thread.
