# Head-Specific Candidate Characterization v1

## Status

Pre-registered before running candidate characterization results.

This is a local mechanistic-interpretability practice experiment. It is not SELF-GROUND, negation-scope work, Gemma work, SAE work, nnsight work, a dashboard, a database, a workbench, or a generic circuit-discovery framework.

## Prior Results Being Characterized

The prior head-specific sweep found narrow replicated candidates on one synthetic generator. The held-out robustness pass then falsified or downgraded those candidates:

- L7H7: `heldout_falsified`
- L9H11: `heldout_falsified`
- L7H11: `heldout_downgraded`
- L7H0: `heldout_falsified`
- L0H8: `heldout_downgraded`

Negative controls and prior raw-attention comparison heads also produced apparent seed-level survival rows. That means the next step is characterization and falsification, not candidate discovery.

## Research Question

Do any fixed heads from the held-out robustness pass have local signatures expected from an induction-like mechanism, or do the signatures fail under attention/effect alignment, position sensitivity, token-domain variation, sequence-length variation, OV diagnostics, QK diagnostics, and counterexample inspection?

## Fixed Candidate Set

Primary heads:

- L7H7
- L9H11
- L7H11
- L7H0
- L0H8

These heads are fixed before characterization. No new head may be added based on characterization results.

## Non-Candidate Comparison Heads

Prior raw-attention comparison heads:

- L0H1
- L0H5
- L0H10
- L11H8
- L0H4

Negative-control heads carried forward from held-out robustness:

- L9H5
- L11H0
- L4H1
- L4H9
- L11H9
- L11H7

These heads are retained to make false positives visible. They are not promoted to candidates based on characterization results.

## Characterization Questions

1. Does attention to the expected source position correlate with causal effect size?
2. Does the head matter at the final destination position, the previous/source position, or both?
3. Does zero/mean ablation reduce the induction metric in the same examples where clean-to-corrupt patching improves it?
4. Does the head show OV copy tendency for expected target tokens?
5. Does QK scoring favor the expected source position over distractors?
6. Does the effect survive token-domain and sequence-length variation?
7. Do negative controls show similar signatures?
8. Are failures structured enough to explain why the candidate should be downgraded?

## Prompt Families

Use existing held-out families:

- `heldout_symbolic_longer`
- `heldout_word_sequences`
- `heldout_number_sequences`
- `heldout_double_repeat`
- `heldout_wrong_target_same_prompt`
- `heldout_no_structure_same_tokens`

Add characterization families:

- `char_symbolic_short`
- `char_symbolic_long`
- `char_word_short`
- `char_word_long`
- `char_number_short`
- `char_number_long`
- `char_multi_distractor`
- `char_reversed_control`
- `char_target_swap_control`

## Metrics

Primary:

- `true_vs_control_logit_diff`

Secondary:

- `target_logit`
- `probability_gap`
- `rank_delta`
- `attention_to_expected_source`
- `attention_to_best_distractor`
- `source_attention_margin`
- `attention_effect_correlation`
- `ablation_patch_consistency`
- `ov_copy_score`
- `qk_source_margin`

## Diagnostics

Attention/effect alignment:

- Compare example-level attention to the expected source position with causal effect size.
- Controls must be evaluated alongside positives.

Position characterization:

- Compare final destination, previous occurrence, source position, and distractor position when metadata supports those positions.
- Unavailable positions must be explicit, not filled in.

OV diagnostic:

- Estimate whether the selected head's OV path promotes the true expected token more than wrong/control tokens.

QK diagnostic:

- Estimate whether the selected head's QK scores favor the expected source position over distractors.

Token-domain and sequence-length characterization:

- Compare symbolic, word, and number domains.
- Compare short and long sequence variants.

## Interventions

- `head_clean_to_corrupt_patch`
- `head_zero_ablation`
- `head_mean_ablation`

All interventions use TransformerLens `hook_z` when head-specific patching is available. If an artifact is not head-specific, it must not be used as head-specific evidence.

## Seeds

Run characterization seeds:

- 20
- 21
- 22

## Decision Rules

`strengthened_local_candidate`:

- survives at least 2 of 3 characterization seeds;
- has positive attention/effect correlation;
- has `source_attention_margin > 0` on positives;
- does not show the same signature on negative controls;
- has consistent direction for clean-to-corrupt patching and at least one ablation;
- has at least weak OV or QK diagnostic support.

`downgraded_candidate`:

- survives only one diagnostic axis;
- survives only one token domain;
- survives only one position/intervention;
- attention/effect alignment is weak;
- ablation and patching disagree;
- negative controls partially match it.

`falsified_candidate`:

- controls match or exceed positives;
- negative controls show the same signatures;
- attention/effect correlation is zero or negative;
- effect disappears under sequence-length or token-domain variation;
- OV/QK diagnostics contradict the story.

## What Counts As Strengthened

A head is strengthened only if it meets the `strengthened_local_candidate` rule across the fixed characterization matrix. Strengthened means "worth manual, example-level study"; it does not mean discovered induction head.

## What Counts As Downgraded

A head is downgraded if evidence is mixed, narrow, position-specific, intervention-specific, token-domain-specific, or matched by negative controls.

## What Counts As Falsified

A head is falsified for this lab stage if controls or negative controls match/exceed it, if attention/effect alignment fails, if position diagnostics contradict the expected role, or if OV/QK diagnostics contradict the local induction-like story.

## What Will Not Be Claimed

This experiment does not prove an induction head.

This experiment does not identify a full circuit.

This experiment does not establish broad GPT-2 behavior.

This experiment can only strengthen, downgrade, or falsify local candidates.

## Run Commands

```bash
uv run python scripts/build_characterization_prompts.py \
  --config configs/gpt2_small_candidate_characterization_seed20.yaml

uv run python scripts/run_candidate_characterization.py \
  --config configs/gpt2_small_candidate_characterization_seed20.yaml \
  --candidate-set reports/head_specific_induction_heldout_robustness_v1/heldout_candidate_set.csv \
  --output reports/head_specific_candidate_characterization_v1/seed20

uv run python scripts/run_candidate_characterization.py \
  --config configs/gpt2_small_candidate_characterization_seed21.yaml \
  --candidate-set reports/head_specific_induction_heldout_robustness_v1/heldout_candidate_set.csv \
  --output reports/head_specific_candidate_characterization_v1/seed21

uv run python scripts/run_candidate_characterization.py \
  --config configs/gpt2_small_candidate_characterization_seed22.yaml \
  --candidate-set reports/head_specific_induction_heldout_robustness_v1/heldout_candidate_set.csv \
  --output reports/head_specific_candidate_characterization_v1/seed22

uv run python scripts/compare_candidate_characterization_runs.py \
  --runs \
    reports/head_specific_candidate_characterization_v1/seed20 \
    reports/head_specific_candidate_characterization_v1/seed21 \
    reports/head_specific_candidate_characterization_v1/seed22 \
  --output reports/head_specific_candidate_characterization_v1
```

## Artifact List

Per seed:

- `prompts.csv`
- `candidate_characterization_results.csv`
- `candidate_characterization_by_candidate.csv`
- `attention_effect_alignment/`
- `position_characterization/`
- `head_circuit_diagnostics/`
- `candidate_characterization_summary.json`
- `candidate_characterization.md`

Consolidated:

- `reports/head_specific_candidate_characterization_v1/run_manifest.json`
- `reports/head_specific_candidate_characterization_v1/candidate_characterization_multiseed_by_candidate.csv`
- `reports/head_specific_candidate_characterization_v1/candidate_characterization_multiseed_by_axis.csv`
- `reports/head_specific_candidate_characterization_v1/candidate_characterization_summary.json`
- `reports/head_specific_candidate_characterization_v1/head_specific_candidate_characterization_v1.md`
- `docs/results/head_specific_candidate_characterization_v1.md` if `reports/` artifacts are ignored
