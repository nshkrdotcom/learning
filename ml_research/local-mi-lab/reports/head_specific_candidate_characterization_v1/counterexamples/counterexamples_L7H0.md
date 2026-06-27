# Candidate Characterization Counterexamples: L7H0

## Why this candidate was inspected

L7H0 was in the fixed primary candidate set from the prior held-out pass. Its final characterization status was `falsified_candidate`.

## Strongest successes

| seed | family | intervention | position_label | effect_size | token_domain |
| --- | --- | --- | --- | --- | --- |
| 21 | char_multi_distractor | head_zero_ablation | final | 0.6247593408836396 | symbolic |
| 21 | char_multi_distractor | head_zero_ablation | previous_occurrence | 0.4657174537851749 | symbolic |
| 21 | char_multi_distractor | head_mean_ablation | final | 0.2703439588625034 | symbolic |
| 20 | char_word_short | head_mean_ablation | final | 0.2413360608006211 | word |
| 21 | char_word_short | head_mean_ablation | final | 0.1812103384946172 | word |
| 21 | char_multi_distractor | head_mean_ablation | previous_occurrence | 0.180981172044375 | symbolic |
| 21 | char_multi_distractor | head_clean_to_corrupt_patch | final | 0.1692971374666939 | symbolic |
| 21 | char_multi_distractor | head_mean_ablation | final | 0.1281933878735388 | symbolic |

## Strongest failures

| seed | family | intervention | position_label | effect_size | token_domain |
| --- | --- | --- | --- | --- | --- |
| 20 | char_multi_distractor | head_zero_ablation | final | -0.1152141308738176 | symbolic |
| 22 | char_multi_distractor | head_zero_ablation | final | -0.1152141308738176 | symbolic |
| 20 | char_multi_distractor | head_zero_ablation | final | -0.101098492350648 | symbolic |
| 22 | char_symbolic_short | head_zero_ablation | final | -0.087828681290821 | symbolic |
| 20 | char_symbolic_short | head_zero_ablation | final | -0.087828681290821 | symbolic |
| 21 | char_multi_distractor | head_clean_to_corrupt_patch | previous_occurrence | -0.0668053157392441 | symbolic |
| 21 | char_multi_distractor | head_zero_ablation | previous_occurrence | -0.0640535848212251 | symbolic |
| 20 | char_number_short | head_zero_ablation | final | -0.0638097868884225 | number |

## Controls that moved

| seed | family | intervention | position_label | effect_size | token_domain |
| --- | --- | --- | --- | --- | --- |
| 22 | char_reversed_control | head_clean_to_corrupt_patch | final | 0.5277449707665127 | symbolic |
| 22 | char_reversed_control | head_zero_ablation | final | 0.4286341763379567 | symbolic |
| 22 | char_reversed_control | head_mean_ablation | final | 0.4276385784423773 | symbolic |
| 22 | char_reversed_control | head_clean_to_corrupt_patch | final | 0.250482102617438 | symbolic |
| 22 | char_reversed_control | head_mean_ablation | final | 0.2083929190500953 | symbolic |
| 21 | char_reversed_control | head_zero_ablation | final | 0.0999107660848684 | symbolic |
| 20 | char_reversed_control | head_zero_ablation | final | 0.0508808161283081 | symbolic |
| 20 | char_target_swap_control | head_clean_to_corrupt_patch | final | 0.039565667829895 | symbolic |

## Negative-control comparison

Top negative-control rows by mean gap: `[{'layer': 4, 'head': 1, 'mean_positive_minus_control_gap': -0.0066721489855266, 'final_characterization_status': 'falsified_candidate'}, {'layer': 11, 'head': 0, 'mean_positive_minus_control_gap': -0.0096445696267726, 'final_characterization_status': 'falsified_candidate'}, {'layer': 11, 'head': 7, 'mean_positive_minus_control_gap': -0.0152257515614516, 'final_characterization_status': 'falsified_candidate'}, {'layer': 11, 'head': 9, 'mean_positive_minus_control_gap': -0.0172443501949977, 'final_characterization_status': 'falsified_candidate'}, {'layer': 9, 'head': 5, 'mean_positive_minus_control_gap': -0.0193007068378529, 'final_characterization_status': 'falsified_candidate'}]`.

## Token domains where it failed

| seed | family | intervention | position_label | effect_size | token_domain |
| --- | --- | --- | --- | --- | --- |
| 20 | char_symbolic_short | head_zero_ablation | final | -0.087828681290821 | symbolic |
| 22 | char_symbolic_short | head_zero_ablation | final | -0.087828681290821 | symbolic |
| 20 | char_number_short | head_zero_ablation | final | -0.0638097868884225 | number |
| 21 | char_number_short | head_zero_ablation | final | -0.0407495453910432 | number |
| 20 | char_word_long | head_zero_ablation | final | -0.0384033037154112 | word |
| 21 | char_word_long | head_zero_ablation | final | -0.0384033037154112 | word |
| 20 | char_number_short | head_clean_to_corrupt_patch | final | -0.0322395337299542 | number |
| 21 | char_symbolic_long | head_mean_ablation | final | -0.0281272354444642 | symbolic |

## Sequence lengths where it failed

| seed | family | intervention | position_label | effect_size | token_domain | sequence_length_bucket |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | char_symbolic_short | head_zero_ablation | final | -0.087828681290821 | symbolic | short |
| 22 | char_symbolic_short | head_zero_ablation | final | -0.087828681290821 | symbolic | short |
| 20 | char_number_short | head_zero_ablation | final | -0.0638097868884225 | number | short |
| 21 | char_number_short | head_zero_ablation | final | -0.0407495453910432 | number | short |
| 20 | char_word_long | head_zero_ablation | final | -0.0384033037154112 | word | long |
| 21 | char_word_long | head_zero_ablation | final | -0.0384033037154112 | word | long |
| 20 | char_number_short | head_clean_to_corrupt_patch | final | -0.0322395337299542 | number | short |
| 21 | char_symbolic_long | head_mean_ablation | final | -0.0281272354444642 | symbolic | long |

## Position/intervention mismatch

| seed | family | intervention | position_label | effect_size | token_domain |
| --- | --- | --- | --- | --- | --- |
| 20 | char_multi_distractor | head_zero_ablation | final | -0.1152141308738176 | symbolic |
| 22 | char_multi_distractor | head_zero_ablation | final | -0.1152141308738176 | symbolic |
| 20 | char_multi_distractor | head_zero_ablation | final | -0.101098492350648 | symbolic |
| 22 | char_symbolic_short | head_zero_ablation | final | -0.087828681290821 | symbolic |
| 20 | char_symbolic_short | head_zero_ablation | final | -0.087828681290821 | symbolic |
| 21 | char_multi_distractor | head_clean_to_corrupt_patch | previous_occurrence | -0.0668053157392441 | symbolic |
| 21 | char_multi_distractor | head_zero_ablation | previous_occurrence | -0.0640535848212251 | symbolic |
| 20 | char_number_short | head_zero_ablation | final | -0.0638097868884225 | number |

## Attention/effect mismatch

Read this alongside the attention/effect alignment artifact. A positive or negative single-example effect is not enough without candidate-level alignment.

## OV/QK mismatch

Read this alongside the OV/QK diagnostic artifact. OV and QK margins are local signatures only, and they did not establish a complete circuit.

## Interpretation

The counterexamples are part of the result. This candidate should not be upgraded unless its failures, moved controls, and position/intervention sensitivity are explained.

## What not to claim

Do not claim induction-head discovery, a circuit, or broad GPT-2 behavior from this artifact.
