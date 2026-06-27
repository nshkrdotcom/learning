# Induction False Positives: GPT-2 Small Practice Note

## Question

Are heads with high previous-occurrence attention on repeated-token prompts specific to induction-like behavior, or do they also score on controls?

## What looked promising before controls

The basic induction run looked clean: GPT-2 small predicted the expected repeated token strongly, and heads like L0H1 and L0H5 showed high previous-occurrence attention. Without controls, those looked like plausible induction-like attention pattern candidates.

## What controls showed

Raw previous-occurrence attention was not enough. The same raw attention heads also scored highly on controls, especially random-expected-token and distractor-style controls. Random-expected-token controls were important because the prompt structure was unchanged while the scored target token was wrong. Distractor controls were important because they tested whether repeated structure could point toward misleading alternatives.

## What controlled patching showed

The controlled patching follow-up patched 11 selected `attn_out` layer-level candidates on four families with eight examples per family. Patching was not head-specific. The overall positive mean effect size was `0.0522`, while the max control mean effect size was `0.1755`. Six candidates met the simple positive-specific rule, but the largest causal gap came from a random comparison head, not from a raw previous-occurrence attention candidate.

## What I Learned

Causal effects must be compared against controls. A candidate can look interesting on positive prompts and still fail the specificity test. A positive-minus-control causal gap is more informative than raw attention, but it still needs replication and manual inspection.

## Mistakes This Prevents

This prevents calling raw attention an induction head. It also prevents treating a layer-level `attn_out` patch as head-specific, and it prevents ignoring controls that move as much as or more than positives.

## What I Would Test Next

Run the same tiny workflow with a new prompt seed because at least one candidate met the simple positive-specific rule. If the random comparison head does not replicate, treat the current largest gap as a small-sample artifact.

## What I Will Not Claim

This was practice, not a publishable induction-head result. I will not claim a mechanism, a circuit, or a head-specific result from this run. I will not claim broad GPT-2 behavior from this selected prompt set and target-logit metric.
