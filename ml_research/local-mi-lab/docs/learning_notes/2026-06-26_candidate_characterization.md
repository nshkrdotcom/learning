# Candidate Characterization: Practice Note

## Question

Can the fixed heads from the held-out robustness pass be strengthened by local signatures that look induction-like, or do the added diagnostics downgrade them?

## Why robustness was still not enough

A candidate can replicate without being mechanistically understood. The held-out pass already showed that apparent survival rows were not specific enough, because negative controls and prior raw-attention comparison heads could also look good under permissive seed-level rules.

## What characterization added

The characterization pass kept the candidate set fixed and added example-level attention/effect alignment, position sensitivity, token-domain and sequence-length variation, and local OV/QK diagnostics. It ran seeds 20, 21, and 22 without selecting new heads from the results.

## Attention/effect lesson

Attention/effect alignment is stronger than attention alone, but still not a circuit proof. In this run, weak positive correlations appeared for some heads, but they did not combine with robust positive-minus-control causal gaps. L7H7, L9H11, L7H11, L7H0, and L0H8 all ended as `falsified_candidate`.

## Position lesson

A candidate that only works for final-position patching may not be doing the hypothesized source-selection role. Position statuses varied across seeds and were not enough to rescue the candidates once controls, causal gaps, and diagnostic axes were included.

## OV/QK lesson

OV and QK diagnostics are local signatures, not a complete circuit. Some heads had local OV or QK support in individual seeds, but this support also appeared in prior raw-attention or negative-control heads. It did not establish a strengthened local induction candidate.

## Token-Domain and Sequence-Length Lesson

A candidate that only survives one token domain or one sequence length is fragile. The characterization prompts included symbolic, word, number, short, long, multi-distractor, reversed-control, and target-swap constructions. The primary heads did not survive this broader local variation.

## Negative-Control Lesson

Negative controls are part of the interpretation, not an appendix. The corrected characterization run found seed-level support for a negative-control head and for a prior raw-attention comparison head. That makes the primary-candidate story weaker, not stronger.

## What Strengthened

No primary candidate strengthened under the multi-seed characterization rule.

## What Downgraded

The earlier replicated and held-out effects were downgraded because the combined evidence did not align across causal effect, attention/effect alignment, position sensitivity, and OV/QK diagnostics.

## What Falsified

All 16 fixed heads in the consolidated characterization table were classified as `falsified_candidate`, including the five primary heads: L7H7, L9H11, L7H11, L7H0, and L0H8.

## What I Learned About MI Practice

The main progress is not preserving a candidate. The progress is that the workflow can make a promising-looking result fail under stricter tests.

## Mistakes This Prevents

This prevents treating raw attention as causal evidence, treating one prompt generator as robust evidence, ignoring negative controls, and mistaking local OV/QK signatures for a full mechanism.

## What I Would Test Next

Before adding models or larger tooling, write a short result note and inspect why the synthetic prompt families create fragile effects. A smaller hand-curated prompt set may be more useful than more candidate heads.

## What I Will Not Claim

No induction-head discovery has been established. No full circuit has been established. No broad GPT-2 behavior has been established. This is a local MI practice result that falsified the current fixed candidates under the characterization rule.
