# Head-Specific Induction Held-Out Robustness v1

## Executive summary

The held-out robustness check falsified or downgraded the previously replicated candidates. They should not be treated as induction-head candidates beyond the original synthetic setup.

This is local MI practice evidence. It is not a mechanism claim, a circuit claim, or broad GPT-2 behavior.

## Prior candidate result

The prior head-specific sweep found narrow replicated candidates on one synthetic prompt generator. L7H7 was especially important to test because it was previously a random-comparison candidate.

## Held-out design

- Model: `gpt2-small`
- Primary metric: `true_vs_control_logit_diff`
- Hook: `blocks.<layer>.attn.hook_z`
- Interventions: clean-to-corrupt patching, zero ablation, and mean ablation
- Positions: final and previous occurrence
- Seeds: 10, 11, 12

## Fixed candidate set

The held-out runs used a fixed candidate set selected from prior artifacts before held-out scoring. No heads were selected from held-out outcomes.

## Prompt families

- heldout_no_structure_same_tokens: mean effect `-0.0305`, type `control`.
- heldout_wrong_target_same_prompt: mean effect `n/a`, type `control`.
- heldout_number_sequences: mean effect `0.0053`, type `positive`.
- heldout_symbolic_longer: mean effect `0.0007`, type `positive`.
- heldout_double_repeat: mean effect `-0.0024`, type `positive`.
- heldout_word_sequences: mean effect `-0.0047`, type `positive`.

## Interventions and positions

Final-position rows were available for all interventions. Previous-occurrence rows were frequently unavailable when the held-out control metadata had no meaningful source position; those rows are reported as insufficient rather than filled in.

## Seed-level results

Status counts: `{'heldout_falsified': 11, 'heldout_downgraded': 5}`

## Candidate-level survival

- L7H11: `heldout_downgraded`, survived seeds `1`, mean gap `0.1829` (replicated_candidate).
- L7H0: `heldout_falsified`, survived seeds `0`, mean gap `0.0744` (replicated_candidate).
- L0H8: `heldout_downgraded`, survived seeds `2`, mean gap `0.0006` (replicated_candidate).
- L7H7: `heldout_falsified`, survived seeds `3`, mean gap `-0.0094` (random_comparison_replicated).
- L9H11: `heldout_falsified`, survived seeds `3`, mean gap `-0.0494` (replicated_candidate).

## L7H7

L7H7 is classified as `heldout_falsified`. It survived 3 seed(s), with mean gap `-0.0094` and survived interventions `head_clean_to_corrupt_patch,head_zero_ablation`. This is not a mechanism claim.

## L9H11

L9H11 is classified as `heldout_falsified`. It survived 3 seed(s), with mean gap `-0.0494` and survived interventions `head_clean_to_corrupt_patch,head_mean_ablation,head_zero_ablation`. This is not a mechanism claim.

## Other replicated candidates

- L7H11: `heldout_downgraded`, survived seeds `1`, mean gap `0.1829` (replicated_candidate).
- L7H0: `heldout_falsified`, survived seeds `0`, mean gap `0.0744` (replicated_candidate).
- L0H8: `heldout_downgraded`, survived seeds `2`, mean gap `0.0006` (replicated_candidate).

## Prior raw-attention heads

- L0H1: `heldout_downgraded`, survived seeds `2`, mean gap `0.1430` (prior_raw_attention_failed).
- L0H10: `heldout_falsified`, survived seeds `2`, mean gap `0.1222` (prior_raw_attention_failed).
- L0H5: `heldout_downgraded`, survived seeds `2`, mean gap `0.0595` (prior_raw_attention_failed).
- L0H4: `heldout_falsified`, survived seeds `2`, mean gap `-0.0144` (prior_raw_attention_failed).
- L11H8: `heldout_falsified`, survived seeds `0`, mean gap `-0.1685` (prior_raw_attention_failed).

## Negative controls

- L11H0: `heldout_falsified`, survived seeds `3`, mean gap `0.1124` (negative_control_no_effect).
- L11H9: `heldout_downgraded`, survived seeds `1`, mean gap `-0.0038` (negative_control_nonspecific).
- L4H1: `heldout_falsified`, survived seeds `0`, mean gap `-0.0040` (negative_control_no_effect).
- L9H5: `heldout_falsified`, survived seeds `1`, mean gap `-0.0056` (negative_control_no_effect).
- L4H9: `heldout_falsified`, survived seeds `2`, mean gap `-0.0080` (negative_control_nonspecific).
- L11H7: `heldout_falsified`, survived seeds `1`, mean gap `-0.0369` (negative_control_nonspecific).

## Counterexamples

Counterexample inspection should focus on candidates that were downgraded or falsified, especially when controls moved or intervention variants disagreed.

## What survived

No candidate cleanly survived the held-out rule.

## What failed

Candidates with controls-moving failures, no-positive-effect rows, intervention-only effects, or unavailable previous-occurrence rows are downgraded or falsified for this lab stage.

## What this teaches

Held-out construction, causal controls, intervention variants, and position variants can all break a result that looked replicated under one generator.

## What this does not show

This does not discover an induction head, identify a circuit, or establish broad GPT-2 behavior. Even a surviving candidate remains local to the tested prompts, metric, hook, intervention, and positions.

## Recommendation

Treat the prior replicated candidates as downgraded or falsified for this lab stage, then write the learning note before adding any new model or task.
