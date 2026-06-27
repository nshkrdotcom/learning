# Head-Specific Candidate Characterization v1

## Executive summary

No prior candidate survived characterization as a strengthened local candidate. The previous head-specific and held-out effects should be treated as fragile local artifacts.

This report refuses induction-head, circuit, and broad GPT-2 claims. It only summarizes a fixed-candidate local characterization pass.

## Prior held-out result

The prior held-out robustness pass produced mixed and nonspecific results: no candidate was safe to treat as an induction-head discovery, and negative controls made the permissive survival rule look too weak.

## Fixed candidates

| head | candidate_id | group | final status | mean gap | mean corr | OV | QK |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L4H1 | heldout_cand_012 | negative_control_no_effect | falsified_candidate | -0.0067 | -0.0640 | ov_supports_copy,ov_weak | qk_weak |
| L9H5 | heldout_cand_010 | negative_control_no_effect | falsified_candidate | -0.0193 | 0.0442 | ov_supports_copy,ov_weak | qk_weak |
| L11H0 | heldout_cand_011 | negative_control_no_effect | falsified_candidate | -0.0096 | 0.0870 | ov_supports_copy | qk_weak |
| L4H9 | heldout_cand_013 | negative_control_nonspecific | falsified_candidate | -0.0203 | -0.1355 | ov_supports_copy,ov_weak | qk_weak |
| L11H7 | heldout_cand_015 | negative_control_nonspecific | falsified_candidate | -0.0152 | -0.0920 | ov_contradicts_copy,ov_weak | qk_weak |
| L11H9 | heldout_cand_014 | negative_control_nonspecific | falsified_candidate | -0.0172 | 0.0892 | ov_contradicts_copy,ov_supports_copy,ov_weak | qk_supports_source_selection,qk_weak |
| L0H1 | heldout_cand_005 | prior_raw_attention_failed | falsified_candidate | -0.3224 | 0.0360 | ov_supports_copy,ov_weak | qk_supports_source_selection |
| L0H4 | heldout_cand_009 | prior_raw_attention_failed | falsified_candidate | 0.0102 | 0.1285 | ov_supports_copy | qk_supports_source_selection |
| L0H5 | heldout_cand_006 | prior_raw_attention_failed | falsified_candidate | -0.0932 | -0.0048 | ov_supports_copy | qk_supports_source_selection |
| L0H10 | heldout_cand_007 | prior_raw_attention_failed | falsified_candidate | -0.2550 | 0.0112 | ov_supports_copy,ov_weak | qk_supports_source_selection |
| L11H8 | heldout_cand_008 | prior_raw_attention_failed | falsified_candidate | -0.1815 | 0.0124 | ov_contradicts_copy,ov_weak | qk_weak |
| L7H7 | heldout_cand_000 | random_comparison_replicated | falsified_candidate | -0.0550 | 0.0370 | ov_supports_copy,ov_weak | qk_weak |
| L0H8 | heldout_cand_004 | replicated_candidate | falsified_candidate | -0.1949 | 0.0416 | ov_weak | qk_weak |
| L7H0 | heldout_cand_003 | replicated_candidate | falsified_candidate | -0.0336 | -0.0179 | ov_weak | qk_weak |
| L7H11 | heldout_cand_002 | replicated_candidate | falsified_candidate | -0.1316 | 0.0436 | ov_supports_copy,ov_weak | qk_weak |
| L9H11 | heldout_cand_001 | replicated_candidate | falsified_candidate | -0.0048 | 0.0390 | ov_supports_copy,ov_weak | qk_supports_source_selection,qk_weak |

## Characterization design

- Primary metric: `true_vs_control_logit_diff`.
- Fixed primary heads: `L7H7`, `L9H11`, `L7H11`, `L7H0`, `L0H8`.
- Prior raw-attention heads and deterministic negative controls were retained.
- No candidates were selected from characterization results.

## Attention/effect alignment

| head | group | mean value | statuses |
| --- | --- | --- | --- |
| L0H4 | prior_raw_attention_failed | 0.1285 | characterization_falsifies,characterization_supports |
| L11H9 | negative_control_nonspecific | 0.0892 | characterization_falsifies |
| L11H0 | negative_control_no_effect | 0.0870 | characterization_falsifies,characterization_supports |
| L9H5 | negative_control_no_effect | 0.0442 | characterization_downgrades,characterization_falsifies |
| L7H11 | replicated_candidate | 0.0436 | characterization_falsifies |
| L0H8 | replicated_candidate | 0.0416 | characterization_falsifies |
| L9H11 | replicated_candidate | 0.0390 | characterization_falsifies |
| L7H7 | random_comparison_replicated | 0.0370 | characterization_falsifies |
| L0H1 | prior_raw_attention_failed | 0.0360 | characterization_falsifies |
| L11H8 | prior_raw_attention_failed | 0.0124 | characterization_falsifies |
| L0H10 | prior_raw_attention_failed | 0.0112 | characterization_falsifies |
| L0H5 | prior_raw_attention_failed | -0.0048 | characterization_falsifies |
| L7H0 | replicated_candidate | -0.0179 | characterization_falsifies |
| L4H1 | negative_control_no_effect | -0.0640 | characterization_falsifies,characterization_supports |
| L11H7 | negative_control_nonspecific | -0.0920 | characterization_falsifies |
| L4H9 | negative_control_nonspecific | -0.1355 | characterization_falsifies |

## Position sensitivity

| head | group | position statuses |
| --- | --- | --- |
| L4H1 | negative_control_no_effect | destination_specific,position_nonspecific,source_specific |
| L9H5 | negative_control_no_effect | position_nonspecific,source_specific |
| L11H0 | negative_control_no_effect | destination_specific,no_position_effect |
| L4H9 | negative_control_nonspecific | destination_specific,no_position_effect,source_specific |
| L11H7 | negative_control_nonspecific | destination_specific,no_position_effect |
| L11H9 | negative_control_nonspecific | no_position_effect |
| L0H1 | prior_raw_attention_failed | both_source_and_destination,source_specific |
| L0H4 | prior_raw_attention_failed | no_position_effect,position_nonspecific |
| L0H5 | prior_raw_attention_failed | position_nonspecific,source_specific |
| L0H10 | prior_raw_attention_failed | both_source_and_destination,position_nonspecific,source_specific |
| L11H8 | prior_raw_attention_failed | destination_specific,no_position_effect |
| L7H7 | random_comparison_replicated | destination_specific,no_position_effect,source_specific |
| L0H8 | replicated_candidate | no_position_effect,position_nonspecific,source_specific |
| L7H0 | replicated_candidate | destination_specific,no_position_effect,position_nonspecific |
| L7H11 | replicated_candidate | destination_specific |
| L9H11 | replicated_candidate | destination_specific,no_position_effect |

## Token-domain and sequence-length sensitivity

Token-domain and sequence-length variation were built into the characterization prompt families. The consolidated status treats candidates with fragile or inconsistent evidence as downgraded or falsified.

## OV diagnostics

| head | group | mean value | statuses |
| --- | --- | --- | --- |
| L9H5 | negative_control_no_effect | 0.1305 | ov_supports_copy,ov_weak |
| L11H0 | negative_control_no_effect | 0.0684 | ov_supports_copy |
| L0H4 | prior_raw_attention_failed | 0.0415 | ov_supports_copy |
| L7H7 | random_comparison_replicated | 0.0414 | ov_supports_copy,ov_weak |
| L0H5 | prior_raw_attention_failed | 0.0183 | ov_supports_copy |
| L4H9 | negative_control_nonspecific | 0.0176 | ov_supports_copy,ov_weak |
| L4H1 | negative_control_no_effect | 0.0036 | ov_supports_copy,ov_weak |
| L9H11 | replicated_candidate | -0.0037 | ov_supports_copy,ov_weak |
| L0H10 | prior_raw_attention_failed | -0.0098 | ov_supports_copy,ov_weak |
| L7H11 | replicated_candidate | -0.0101 | ov_supports_copy,ov_weak |
| L0H1 | prior_raw_attention_failed | -0.0132 | ov_supports_copy,ov_weak |
| L7H0 | replicated_candidate | -0.0403 | ov_weak |
| L0H8 | replicated_candidate | -0.0477 | ov_weak |
| L11H7 | negative_control_nonspecific | -0.0597 | ov_contradicts_copy,ov_weak |
| L11H9 | negative_control_nonspecific | -0.0677 | ov_contradicts_copy,ov_supports_copy,ov_weak |
| L11H8 | prior_raw_attention_failed | -0.1791 | ov_contradicts_copy,ov_weak |

## QK diagnostics

| head | group | mean value | statuses |
| --- | --- | --- | --- |
| L0H5 | prior_raw_attention_failed | 0.1369 | qk_supports_source_selection |
| L0H1 | prior_raw_attention_failed | 0.1117 | qk_supports_source_selection |
| L0H10 | prior_raw_attention_failed | 0.0228 | qk_supports_source_selection |
| L0H4 | prior_raw_attention_failed | 0.0140 | qk_supports_source_selection |
| L9H11 | replicated_candidate | -0.0024 | qk_supports_source_selection,qk_weak |
| L7H11 | replicated_candidate | -0.0038 | qk_weak |
| L11H7 | negative_control_nonspecific | -0.0067 | qk_weak |
| L11H9 | negative_control_nonspecific | -0.0068 | qk_supports_source_selection,qk_weak |
| L7H7 | random_comparison_replicated | -0.0073 | qk_weak |
| L9H5 | negative_control_no_effect | -0.0081 | qk_weak |
| L11H0 | negative_control_no_effect | -0.0155 | qk_weak |
| L0H8 | replicated_candidate | -0.0181 | qk_weak |
| L7H0 | replicated_candidate | -0.0238 | qk_weak |
| L11H8 | prior_raw_attention_failed | -0.0280 | qk_weak |
| L4H1 | negative_control_no_effect | -0.0305 | qk_weak |
| L4H9 | negative_control_nonspecific | -0.0345 | qk_weak |

## Primary candidates

Outcome counts: `{'falsified_candidate': 5}`.

| head | candidate_id | group | final status | mean gap | mean corr | OV | QK |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L7H7 | heldout_cand_000 | random_comparison_replicated | falsified_candidate | -0.0550 | 0.0370 | ov_supports_copy,ov_weak | qk_weak |
| L0H8 | heldout_cand_004 | replicated_candidate | falsified_candidate | -0.1949 | 0.0416 | ov_weak | qk_weak |
| L7H0 | heldout_cand_003 | replicated_candidate | falsified_candidate | -0.0336 | -0.0179 | ov_weak | qk_weak |
| L7H11 | heldout_cand_002 | replicated_candidate | falsified_candidate | -0.1316 | 0.0436 | ov_supports_copy,ov_weak | qk_weak |
| L9H11 | heldout_cand_001 | replicated_candidate | falsified_candidate | -0.0048 | 0.0390 | ov_supports_copy,ov_weak | qk_supports_source_selection,qk_weak |

## Prior raw-attention comparison heads

Outcome counts: `{'falsified_candidate': 5}`.

| head | candidate_id | group | final status | mean gap | mean corr | OV | QK |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L0H1 | heldout_cand_005 | prior_raw_attention_failed | falsified_candidate | -0.3224 | 0.0360 | ov_supports_copy,ov_weak | qk_supports_source_selection |
| L0H4 | heldout_cand_009 | prior_raw_attention_failed | falsified_candidate | 0.0102 | 0.1285 | ov_supports_copy | qk_supports_source_selection |
| L0H5 | heldout_cand_006 | prior_raw_attention_failed | falsified_candidate | -0.0932 | -0.0048 | ov_supports_copy | qk_supports_source_selection |
| L0H10 | heldout_cand_007 | prior_raw_attention_failed | falsified_candidate | -0.2550 | 0.0112 | ov_supports_copy,ov_weak | qk_supports_source_selection |
| L11H8 | heldout_cand_008 | prior_raw_attention_failed | falsified_candidate | -0.1815 | 0.0124 | ov_contradicts_copy,ov_weak | qk_weak |

## Negative controls

Outcome counts: `{'falsified_candidate': 6}`.

| head | candidate_id | group | final status | mean gap | mean corr | OV | QK |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L4H1 | heldout_cand_012 | negative_control_no_effect | falsified_candidate | -0.0067 | -0.0640 | ov_supports_copy,ov_weak | qk_weak |
| L9H5 | heldout_cand_010 | negative_control_no_effect | falsified_candidate | -0.0193 | 0.0442 | ov_supports_copy,ov_weak | qk_weak |
| L11H0 | heldout_cand_011 | negative_control_no_effect | falsified_candidate | -0.0096 | 0.0870 | ov_supports_copy | qk_weak |
| L4H9 | heldout_cand_013 | negative_control_nonspecific | falsified_candidate | -0.0203 | -0.1355 | ov_supports_copy,ov_weak | qk_weak |
| L11H7 | heldout_cand_015 | negative_control_nonspecific | falsified_candidate | -0.0152 | -0.0920 | ov_contradicts_copy,ov_weak | qk_weak |
| L11H9 | heldout_cand_014 | negative_control_nonspecific | falsified_candidate | -0.0172 | 0.0892 | ov_contradicts_copy,ov_supports_copy,ov_weak | qk_supports_source_selection,qk_weak |

## Counterexamples

Counterexample inspection artifacts should be read before treating any candidate as meaningful. Negative controls and failed prompt families are part of the result.

## Final statuses

`{'falsified_candidate': 16}`

## What strengthened

No candidate strengthened under the local characterization rule.

## What downgraded

None.

## What falsified

L0H4, L9H11, L4H1, L11H0, L11H7, L11H9, L9H5, L4H9, L7H0, L7H7, L0H5, L7H11, L11H8, L0H8, L0H10, L0H1.

## What this teaches

A candidate can replicate under one generator and still fail local characterization. Attention/effect alignment, position sensitivity, OV/QK diagnostics, and negative controls make the pipeline harder to fool.

## What this does not show

This does not show an induction head, a circuit, or a broad GPT-2 property.

## Next recommendation

Write up the negative or downgraded result before adding new heads or new models.
