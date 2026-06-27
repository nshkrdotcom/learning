# Held-Out Induction Robustness: Practice Note

## Question

Do the replicated GPT-2 small head-specific candidates from the first synthetic prompt generator survive held-out prompt constructions, intervention variants, and position variants?

## Why the prior replicated result was not enough

A candidate that survives one synthetic prompt generator is not robust by default. The prior run used seeds 0, 1, and 2, but those seeds still came from the same generator design and the same final-position clean-to-corrupt patching setup.

L7H7 needed extra skepticism because it was originally a random-comparison candidate. A random-comparison candidate needs extra skepticism even when it replicates under a narrow rule.

## What held-out prompt families tested

The held-out pass used longer symbolic sequences, word sequences, number sequences, double-repeat prompts, wrong-target same-prompt controls, and no-structure same-token controls.

Held-out prompt construction is different from merely changing the random seed. It changes the shape of the examples and adds counterexamples that the original generator did not emphasize.

## What intervention variants tested

The held-out pass tested head-specific `hook_z` clean-to-corrupt patching, zero ablation, and mean ablation.

Intervention variants matter because clean-to-corrupt patching and ablation answer different questions. A candidate that only works under one intervention is weaker than one that survives multiple causal checks.

## What survived

No candidate cleanly survived the held-out robustness rule.

L0H8 was downgraded: it survived two seeds under final-position rows, but the mean gap was near zero and the effect did not survive the position-variant requirement.

Some prior raw-attention comparison heads produced apparent seed-level survival rows. That is not a rescue of raw attention; it is another warning that permissive criteria can create false positives.

## What failed

L7H7 was falsified for this lab stage. It had strong positive examples, but controls also moved, and its aggregate held-out status was `heldout_falsified`.

L9H11 was falsified for this lab stage. It had seed-level survival rows, but controls-moving and negative aggregate gaps made the result fail the robustness rule.

L7H11 was downgraded. It had a strong seed-10 clean-to-corrupt result but did not replicate cleanly across held-out seeds and variants.

L7H0 was falsified. It did not produce a useful held-out positive effect.

## What controls revealed

Controls must be causal controls, not just descriptive controls. The no-structure same-token controls and wrong-target controls showed that some interventions can move logits in ways that are not specific to the intended induction structure.

The negative-control heads were also useful. Some produced apparent survival rows, which made it easier to see that seed-level survival was too permissive by itself.

## What I Learned About Robustness

The useful result is not that a candidate was preserved. The useful result is that the robustness pass made the earlier candidate set harder to trust.

Position matters because final-position patching may not test the same causal role as source/previous-position intervention. In this run, previous-occurrence rows were often unavailable or insufficient, so final-only effects were downgraded rather than treated as robust.

A negative held-out result is useful. It prevents treating a local synthetic-generator result as a stronger MI finding.

## Mistakes This Prevents

- Treating multi-seed replication on one generator as held-out robustness.
- Treating a random-comparison candidate as meaningful without extra falsification.
- Trusting final-position clean-to-corrupt patching when ablations or position variants disagree.
- Ignoring controls that move under the same intervention.
- Reporting a candidate as robust when only one intervention or position supports it.

## What I Would Test Next

Before adding a new model or method, I would inspect the failed and downgraded cases manually: why no-structure controls moved, why L7H7 has strong individual successes but fails the aggregate rule, and whether the previous-occurrence position metadata should be improved for controls where a meaningful source exists.

If continuing with induction practice, I would build a smaller prompt set with clearer previous/source-position semantics before running any wider sweep.

## What I Will Not Claim

I will not claim induction-head discovery.

I will not claim a circuit.

I will not claim broad GPT-2 behavior.

I will not claim that L7H7, L9H11, L7H11, L7H0, or L0H8 survived held-out robustness.

I will not claim that raw attention or one synthetic prompt generator is enough.
