# Abstract Drafts

Keep multiple phrasings around and pick one only once Phase 3 results
exist. Don't let the more exciting-sounding one win on vibes — pick the one
that best matches the claim ladder status at submission time.

## Version A — general / broad-audience framing

See `paper/draft.md` "Working abstract" — this is the primary draft.
Leads with the example/label overinterpretation problem, names the harness,
states the case study, and ends with an explicit disclaimer list.

## Version B — MI-reviewer-facing / technical framing

> We present SELF-GROUND, an evaluation harness for testing whether
> candidate sparse-autoencoder (SAE) feature interpretations survive
> controlled decoded activation intervention. Existing practice often
> assigns feature semantics from top-activating examples or short natural-
> language labels, which is known to be vulnerable to descriptive collision
> and does not establish a causal role. SELF-GROUND verifies semantic,
> shape, and reconstruction compatibility between a model checkpoint, hook
> point, and SAELens release/id (failing closed on checkpoint-metadata
> mismatch even when tensor shapes coincide); ranks SAE features by a
> deterministic activation-contrast score over matched minimal pairs;
> performs real encode -> ablate/amplify -> decode -> patch -> measure
> interventions; and compares the resulting target-prompt logit-contrast
> effects against deterministic random and bottom-active control feature
> sets and matched non-negation control prompts. Results are reported
> through a thresholded evidence ladder — blocked, insufficient evidence,
> candidate evidence, or strong candidate evidence — that a small smoke-
> scale run cannot satisfy by construction. As a case study, we apply this
> harness to negation-sensitive features in a small open-weight
> transformer, reporting [PLACEHOLDER: per-status outcome once Phase 3 has
> run]. We do not claim broad model understanding of negation, monosemantic
> feature discovery, or causal mechanism discovery beyond the narrowly
> scoped claim the evidence supports.

## Version C — harness-first, results-agnostic framing (use if Phase 3
results come back `insufficient_evidence` or `blocked`)

> Evaluating whether a sparse-autoencoder feature plays a specific causal
> role typically relies on example inspection or natural-language labels,
> neither of which is falsifiable in the way a causal claim should be. We
> describe SELF-GROUND, a reproducible scaffold that turns a candidate SAE
> feature interpretation into a testable prediction: that intervening on
> the feature (ablation or amplification, decoded back to residual space
> and patched into a real forward pass) will move a matched target prompt
> more than it moves a matched non-negation control prompt, and more than a
> deterministic control feature set from the same ranking. We report the
> infrastructure, the fail-closed compatibility checks needed before any
> intervention can be trusted, and a first negation-scope case study,
> [PLACEHOLDER: whose evidence currently sits at <status> — recording this
> honestly is itself part of the contribution: a harness that can return
> "not enough evidence" is more useful than one that cannot].

## Notes on choosing

- If Phase 3 reaches `candidate_evidence` or better: Version A or B, with
  the placeholder in B filled in honestly (don't round up to
  `strong_candidate_evidence` language unless the thresholds were actually
  met — see D006).
- If Phase 3 lands at `insufficient_evidence` or `blocked`: Version C. A
  paper that honestly reports "the harness works, the first case study's
  evidence wasn't strong enough to clear the candidate threshold, here's
  why" is a legitimate methods contribution and is *more* credible than
  quietly re-running until something looks better.
