# Head-Specific Induction Held-Out Robustness v1

## Status

Pre-registered before held-out prompt generation, held-out scoring, or held-out candidate reclassification.

## Prior result being tested

The prior head-specific run found narrow replicated candidates under GPT-2 small, selected synthetic prompt families, final-position `hook_z` clean-to-corrupt patching, and `true_vs_control_logit_diff`. The strongest candidate, L7H7, was flagged as a prior random-comparison head. This held-out robustness pass tests whether those candidates survive prompt-generator changes and intervention variants.

## Research question

Do the replicated head-specific candidates survive held-out prompt constructions, intervention variants, and stricter counterexamples, or were they artifacts of the original synthetic prompt generator and final-position clean-to-corrupt metric?

## Fixed candidates

Primary candidates:

- L7H7
- L9H11
- L7H11
- L7H0
- L0H8

Prior raw-attention heads for comparison:

- L0H1
- L0H5
- L0H10
- L11H8
- L0H4

Negative-control heads are chosen deterministically from non-replicated `no_effect` and `nonspecific` rows in `reports/head_specific_induction_causality_v1/head_specific_multiseed_by_head.csv`. They are selected before held-out scoring and must not be chosen from held-out results.

## Negative-Control Heads

The candidate-selection artifact must include at least five deterministic negative controls split across:

- `negative_control_no_effect`
- `negative_control_nonspecific`

If enough controls cannot be selected from the prior report, the selector should fail clearly.

## Prompt families

Held-out prompt families are not just random-seed variants of the original generator. They test longer contexts, different token domains, and controls designed to break target-specific or structure-specific artifacts.

Implemented in the main pass:

- `heldout_symbolic_longer`: longer symbolic sequences, for example `A B C D E F A B C D E`, expected `F`.
- `heldout_word_sequences`: common-word sequences, for example `red blue green yellow red blue green`, expected `yellow`.
- `heldout_number_sequences`: number-word sequences, for example `one two three four one two three`, expected `four`.
- `heldout_double_repeat`: two repeated subsequences where only one licenses the target, for example `A B C X Y A B C X`, expected `Y`.
- `heldout_wrong_target_same_prompt`: same prompt as a positive example but scored against a wrong target token.
- `heldout_no_structure_same_tokens`: same token bag as a positive example without an ordered repeated-prefix structure.

Additional prompt families may be prototyped later, but they are not part of the main decision rule for this pass.

## Metrics

Primary metric:

- `true_vs_control_logit_diff`

Also report:

- `target_logit`
- `probability_gap`
- `rank_delta`

Only `true_vs_control_logit_diff` is used for survival decisions in this experiment.

## Interventions

Required:

- `head_clean_to_corrupt_patch`
- `head_zero_ablation`

Optional if cleanly supported:

- `head_mean_ablation`

Mean ablation should replace the selected head and position with the mean activation from selected clean prompts for the same layer, head, and position. If this becomes too large or unclear, it must be recorded as unsupported rather than silently skipped.

## Position variants

Required:

- `final`
- `previous_occurrence`

`previous_occurrence` uses prompt metadata. Controls without a meaningful previous occurrence must be marked explicitly as unavailable for that position. Do not invent source positions for controls.

## Seeds

Held-out seeds:

- 10
- 11
- 12

These differ from the original seeds 0, 1, and 2 so held-out prompt construction is not just a repeat of the prior prompt set.

## Decision rules

A candidate survives only if the same head:

- has positive-minus-control gap greater than 0 under `true_vs_control_logit_diff`;
- survives at least 2 of 3 held-out seeds;
- survives on at least 2 held-out positive prompt families;
- does not have controls moving as much as positives;
- has consistent effect direction for clean-to-corrupt patching.

A candidate is downgraded if:

- it survives only one seed;
- it survives only one prompt family;
- it survives only `target_logit` but not `true_vs_control_logit_diff`;
- it survives patching but not ablation;
- it survives only final position but not previous/source position.

A candidate is falsified for this lab stage if:

- controls move as much as positives;
- held-out prompt families erase the effect;
- the result flips sign across seeds;
- only random or wrong-target controls produce the effect.

## What counts as survival

`heldout_replicated` means the candidate passed the decision rule above. This is a local held-out robustness result, not a mechanism claim.

## What counts as downgrade

`heldout_downgraded` means the candidate retains some positive evidence but is too intervention-specific, family-specific, seed-specific, or position-specific to carry forward as robust.

## What counts as falsification

`heldout_falsified` means the held-out controls, prompt families, or intervention variants remove or reverse the effect enough that the candidate should not be treated as robust beyond the original setup.

## What will not be claimed

This experiment will not claim induction-head discovery, a full circuit, or broad GPT-2 behavior. Even a surviving candidate is only a local prompt-set candidate requiring deeper manual and benchmark checks.

## Run commands

```bash
uv run python scripts/build_heldout_induction_prompts.py \
  --config configs/gpt2_small_induction_heldout_seed10.yaml

uv run python scripts/select_heldout_robustness_candidates.py \
  --multiseed reports/head_specific_induction_causality_v1/head_specific_multiseed_by_head.csv \
  --output reports/head_specific_induction_heldout_robustness_v1/heldout_candidate_set.csv

uv run python scripts/run_heldout_robustness.py \
  --config configs/gpt2_small_induction_heldout_seed10.yaml \
  --candidate-set reports/head_specific_induction_heldout_robustness_v1/heldout_candidate_set.csv
```

Repeat for seeds 11 and 12, then compare:

```bash
uv run python scripts/compare_heldout_robustness_runs.py \
  --runs runs/<heldout_seed10_run> runs/<heldout_seed11_run> runs/<heldout_seed12_run> \
  --output reports/head_specific_induction_heldout_robustness_v1
```

## Artifact list

- `data/induction_heldout_seed10/prompts.csv`
- `data/induction_heldout_seed11/prompts.csv`
- `data/induction_heldout_seed12/prompts.csv`
- `reports/head_specific_induction_heldout_robustness_v1/heldout_candidate_set.csv`
- `runs/<heldout_run>/heldout_robustness_results.csv`
- `runs/<heldout_run>/heldout_robustness_by_family.csv`
- `runs/<heldout_run>/heldout_robustness_by_candidate.csv`
- `runs/<heldout_run>/heldout_robustness_summary.json`
- `reports/head_specific_induction_heldout_robustness_v1/heldout_multiseed_by_candidate.csv`
- `reports/head_specific_induction_heldout_robustness_v1/heldout_multiseed_summary.json`
- `reports/head_specific_induction_heldout_robustness_v1/head_specific_induction_heldout_robustness_v1.md`
- `reports/head_specific_induction_heldout_robustness_v1/counterexamples_L7H7.md`
- `reports/head_specific_induction_heldout_robustness_v1/counterexamples_L9H11.md`
- `reports/head_specific_induction_heldout_robustness_v1/counterexamples_L7H11.md`
