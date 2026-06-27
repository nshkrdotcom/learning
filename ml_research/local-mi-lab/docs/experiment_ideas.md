# Experiment Ideas

These are beginner-safe, local-only ideas ranked by suitability.

## Start Here

1. GPT-2 small tokenization and logit basics.
2. Logit lens on simple prompts.
3. Induction-head inspection.
4. Activation patching on tiny clean/corrupt examples.

## Next

5. Sentiment contrast with strong controls.
6. Factual recall contrast.
7. Subject/object tracking.
8. Copy/repetition behavior.

## Later

9. SAE feature inspection with SAELens.
10. Gemma-2-2B capability check.
11. nnsight local backend comparison.
12. Small original method comparison.

Do not start with negation-scope.
Do not start with Gemma-2-2B.
Do not start with SAEs.
Do not start with a new framework.

Attention inspection is descriptive. A high previous-occurrence attention score is an induction-like attention pattern candidate, not a mechanism claim. Tiny clean/corrupt patching is practice work, not a full IOI experiment.

For Stage 5, use the induction controls config before inventing a new task. The goal of controls is to catch false positives. Raw attention to a previous token is not enough. A candidate head should separate positives from controls. Failure is useful: if controls also score highly, the current prompt/metric is not specific.

For the Stage 6-lite follow-up, select a small candidate list from the controlled attention artifacts and patch only those sites on a balanced subset of positives and high-scoring controls. A candidate that moves controls as much as positives is nonspecific. A positive-minus-control causal gap is more useful than raw attention, but it is still practice evidence, not an induction-head discovery.
