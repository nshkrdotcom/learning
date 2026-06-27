# Head-Specific Induction Causality: Practice Note

## Question

Can GPT-2 small repeated-token behavior be causally linked to specific attention heads when the intervention is truly head-specific, the metric separates true from control tokens, and the result is checked across seeds?

## Why the previous result was not enough

The earlier controlled patching pass used full layer-level `attn_out` patching. Layer-level `attn_out` patching is not head-specific. It also used a weaker target-logit-style metric, and the largest apparent causal gaps came from random comparison heads rather than the raw previous-occurrence attention heads.

Raw attention is descriptive, not causal. L0H1, L0H5, and L0H10 looked strong on repeated-token positives, but controls showed that raw previous-occurrence attention was not target-specific.

## What changed in this experiment

This pass used `blocks.<layer>.attn.hook_z`, which exposes a head axis for GPT-2 small. The artifacts recorded `head_specific_patch=true` and `actual_patch_scope=single_head_z`.

The primary metric changed to `true_vs_control_logit_diff`. Target-logit movement is weaker than a true-vs-control logit-diff metric because it can reward movement that is not specific to the intended induction target.

The sweep compared positives against `distractor_repeat_control`, `random_expected_token_control`, and `same_token_frequency_control` examples in the causal phase, not only in the descriptive phase. Controls must be included in the causal comparison, not only in the descriptive phase.

## Hook-specific lesson

TransformerLens supports head-specific `hook_z` patching for GPT-2 small in this environment. That changed the interpretation of the intervention: this run patched a selected head output at the selected position rather than a full attention layer output.

That does not make the result a circuit claim. It only means the intervention target was narrower and correctly recorded.

## Metric-specific lesson

The stricter metric reduced the ambiguity from target-logit-only movement. A head had to move the true-vs-control logit difference more on positives than on controls to be interesting.

The replicated candidates were small-effect local candidates, not broad behavior explanations.

## Control-specific lesson

Most original raw-attention candidates still failed. L0H1, L0H5, and L0H10 were no-effect or nonspecific across the multi-seed comparison. L11H8 had positive seeds but was classified as nonspecific because controls also moved.

The strongest replicated candidate, L7H7, was flagged as a prior random-comparison candidate. That makes it more important to inspect manually before treating it as anything more than an interesting local result.

## Replication lesson

One seed is not enough. Across seeds 0, 1, and 2, five narrow candidates met the current replicated-head-specific rule: L7H7, L9H11, L7H11, L7H0, and L0H8.

This is stronger than the prior false-positive result, but still narrow. The prompt set is synthetic, the sweep used selected layers, the intervention was final-position clean-to-corrupt `hook_z` patching, and the comparison controls are simple practice controls.

## What I learned about MI practice

The useful pattern was not "find high attention, then declare a head." The useful pattern was:

1. Build controls that can expose false positives.
2. Use a metric that matches the causal question.
3. Verify the hook actually targets the intended component.
4. Compare causal effects on positives against controls.
5. Replicate before taking a candidate seriously.

A negative result is still a useful MI practice result. In this case, the raw attention candidates mostly stayed negative or nonspecific, while a different small candidate set became worth manual inspection.

## Mistakes this prevents

- Calling layer-level `attn_out` patching head-specific.
- Treating raw attention as causal evidence.
- Trusting target-logit movement without a control-token comparison.
- Running controls only descriptively and leaving them out of patching.
- Treating one seed as enough.
- Treating a random-comparison head as induction evidence without manual inspection.

## What I would test next

Manually inspect L7H7, L9H11, and L7H11 on a small set of positive and control examples. Check attention patterns, clean/corrupt prompts, effect-size failures, and whether the effect survives a held-out prompt construction.

If L7H7 remains strongest, explain why it should no longer be treated as only a random-comparison artifact before any stronger claim.

## What I will not claim

I will not claim an induction-head discovery.

I will not claim a full circuit.

I will not claim this generalizes beyond GPT-2 small, these prompt families, selected layers, final-position `hook_z` patching, and this metric.

I will not claim raw attention is enough.
