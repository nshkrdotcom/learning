# Candidate Characterization Counterexamples: L0H8

## Why this candidate was inspected

L0H8 was in the fixed primary candidate set from the prior held-out pass. Its final characterization status was `falsified_candidate`.

## Strongest successes

| seed | family | intervention | position_label | effect_size | token_domain |
| --- | --- | --- | --- | --- | --- |
| 20 | char_word_short | head_mean_ablation | final | 0.5126364355785167 | word |
| 21 | char_word_short | head_mean_ablation | final | 0.4805787406865453 | word |
| 21 | char_multi_distractor | head_clean_to_corrupt_patch | final | 0.4765431740677973 | symbolic |
| 21 | char_word_short | head_zero_ablation | final | 0.4451000476754669 | word |
| 20 | char_number_short | head_zero_ablation | previous_occurrence | 0.4024327730341773 | number |
| 20 | char_number_short | head_zero_ablation | final | 0.3967985748862188 | number |
| 20 | char_word_long | head_zero_ablation | final | 0.3684092867878529 | word |
| 20 | char_multi_distractor | head_zero_ablation | final | 0.3655095622937911 | symbolic |

## Strongest failures

| seed | family | intervention | position_label | effect_size | token_domain |
| --- | --- | --- | --- | --- | --- |
| 21 | char_multi_distractor | head_zero_ablation | final | -3.7637761828843783 | symbolic |
| 21 | char_multi_distractor | head_zero_ablation | previous_occurrence | -1.3752958245298528 | symbolic |
| 21 | char_multi_distractor | head_zero_ablation | final | -1.2192680774187132 | symbolic |
| 20 | char_multi_distractor | head_zero_ablation | final | -1.1527696166173866 | symbolic |
| 22 | char_multi_distractor | head_zero_ablation | final | -1.1527696166173866 | symbolic |
| 22 | char_multi_distractor | head_zero_ablation | final | -0.6716271793581559 | symbolic |
| 21 | char_multi_distractor | head_mean_ablation | final | -0.6158835984536146 | symbolic |
| 20 | char_multi_distractor | head_zero_ablation | previous_occurrence | -0.4056552410730258 | symbolic |

## Controls that moved

| seed | family | intervention | position_label | effect_size | token_domain |
| --- | --- | --- | --- | --- | --- |
| 20 | char_target_swap_control | head_zero_ablation | final | 3.9638556118503088 | symbolic |
| 22 | char_reversed_control | head_zero_ablation | final | 2.329712146819206 | symbolic |
| 20 | char_target_swap_control | head_mean_ablation | final | 1.3437189260855154 | symbolic |
| 21 | char_reversed_control | head_zero_ablation | final | 0.5033469222103996 | symbolic |
| 20 | char_reversed_control | head_zero_ablation | final | 0.2100904040781478 | symbolic |
| 20 | char_target_swap_control | head_zero_ablation | final | 0.1971228789402819 | symbolic |
| 22 | char_reversed_control | head_mean_ablation | final | 0.1854849193967758 | symbolic |
| 20 | char_reversed_control | head_zero_ablation | final | 0.1497878090003429 | symbolic |

## Negative-control comparison

Top negative-control rows by mean gap: `[{'layer': 4, 'head': 1, 'mean_positive_minus_control_gap': -0.0066721489855266, 'final_characterization_status': 'falsified_candidate'}, {'layer': 11, 'head': 0, 'mean_positive_minus_control_gap': -0.0096445696267726, 'final_characterization_status': 'falsified_candidate'}, {'layer': 11, 'head': 7, 'mean_positive_minus_control_gap': -0.0152257515614516, 'final_characterization_status': 'falsified_candidate'}, {'layer': 11, 'head': 9, 'mean_positive_minus_control_gap': -0.0172443501949977, 'final_characterization_status': 'falsified_candidate'}, {'layer': 9, 'head': 5, 'mean_positive_minus_control_gap': -0.0193007068378529, 'final_characterization_status': 'falsified_candidate'}]`.

## Token domains where it failed

| seed | family | intervention | position_label | effect_size | token_domain |
| --- | --- | --- | --- | --- | --- |
| 21 | char_multi_distractor | head_zero_ablation | final | -3.7637761828843783 | symbolic |
| 21 | char_multi_distractor | head_zero_ablation | previous_occurrence | -1.3752958245298528 | symbolic |
| 21 | char_multi_distractor | head_zero_ablation | final | -1.2192680774187132 | symbolic |
| 22 | char_multi_distractor | head_zero_ablation | final | -1.1527696166173866 | symbolic |
| 20 | char_multi_distractor | head_zero_ablation | final | -1.1527696166173866 | symbolic |
| 22 | char_multi_distractor | head_zero_ablation | final | -0.6716271793581559 | symbolic |
| 21 | char_multi_distractor | head_mean_ablation | final | -0.6158835984536146 | symbolic |
| 20 | char_multi_distractor | head_zero_ablation | previous_occurrence | -0.4056552410730258 | symbolic |

## Sequence lengths where it failed

| seed | family | intervention | position_label | effect_size | token_domain | sequence_length_bucket |
| --- | --- | --- | --- | --- | --- | --- |
| 21 | char_multi_distractor | head_zero_ablation | final | -3.7637761828843783 | symbolic | medium |
| 21 | char_multi_distractor | head_zero_ablation | previous_occurrence | -1.3752958245298528 | symbolic | medium |
| 21 | char_multi_distractor | head_zero_ablation | final | -1.2192680774187132 | symbolic | medium |
| 22 | char_multi_distractor | head_zero_ablation | final | -1.1527696166173866 | symbolic | medium |
| 20 | char_multi_distractor | head_zero_ablation | final | -1.1527696166173866 | symbolic | medium |
| 22 | char_multi_distractor | head_zero_ablation | final | -0.6716271793581559 | symbolic | medium |
| 21 | char_multi_distractor | head_mean_ablation | final | -0.6158835984536146 | symbolic | medium |
| 20 | char_multi_distractor | head_zero_ablation | previous_occurrence | -0.4056552410730258 | symbolic | medium |

## Position/intervention mismatch

| seed | family | intervention | position_label | effect_size | token_domain |
| --- | --- | --- | --- | --- | --- |
| 21 | char_multi_distractor | head_zero_ablation | final | -3.7637761828843783 | symbolic |
| 21 | char_multi_distractor | head_zero_ablation | final | -1.2192680774187132 | symbolic |
| 20 | char_multi_distractor | head_zero_ablation | final | -1.1527696166173866 | symbolic |
| 22 | char_multi_distractor | head_zero_ablation | final | -1.1527696166173866 | symbolic |
| 22 | char_multi_distractor | head_zero_ablation | final | -0.6716271793581559 | symbolic |
| 21 | char_multi_distractor | head_mean_ablation | final | -0.6158835984536146 | symbolic |
| 21 | char_multi_distractor | head_mean_ablation | final | -0.3995189575943466 | symbolic |
| 20 | char_multi_distractor | head_zero_ablation | final | -0.3913578389501475 | symbolic |

## Attention/effect mismatch

Read this alongside the attention/effect alignment artifact. A positive or negative single-example effect is not enough without candidate-level alignment.

## OV/QK mismatch

Read this alongside the OV/QK diagnostic artifact. OV and QK margins are local signatures only, and they did not establish a complete circuit.

## Interpretation

The counterexamples are part of the result. This candidate should not be upgraded unless its failures, moved controls, and position/intervention sensitivity are explained.

## What not to claim

Do not claim induction-head discovery, a circuit, or broad GPT-2 behavior from this artifact.
