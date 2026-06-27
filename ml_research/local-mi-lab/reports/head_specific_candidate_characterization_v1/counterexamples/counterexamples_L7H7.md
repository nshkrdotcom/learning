# Candidate Characterization Counterexamples: L7H7

## Why this candidate was inspected

L7H7 was in the fixed primary candidate set from the prior held-out pass. Its final characterization status was `falsified_candidate`.

## Strongest successes

| seed | family | intervention | position_label | effect_size | token_domain |
| --- | --- | --- | --- | --- | --- |
| 21 | char_multi_distractor | head_zero_ablation | final | 0.6623931859681255 | symbolic |
| 21 | char_multi_distractor | head_mean_ablation | final | 0.6610306002548643 | symbolic |
| 21 | char_multi_distractor | head_clean_to_corrupt_patch | final | 0.3771738274694721 | symbolic |
| 21 | char_multi_distractor | head_mean_ablation | final | 0.3556085473839486 | symbolic |
| 21 | char_multi_distractor | head_zero_ablation | final | 0.3048496500743773 | symbolic |
| 21 | char_multi_distractor | head_clean_to_corrupt_patch | final | 0.2519349268782444 | symbolic |
| 20 | char_multi_distractor | head_clean_to_corrupt_patch | final | 0.1038181545908188 | symbolic |
| 20 | char_symbolic_short | head_zero_ablation | final | 0.0945045484178628 | symbolic |

## Strongest failures

| seed | family | intervention | position_label | effect_size | token_domain |
| --- | --- | --- | --- | --- | --- |
| 20 | char_word_short | head_zero_ablation | final | -0.3867558568378301 | word |
| 21 | char_word_short | head_zero_ablation | final | -0.3867558568378301 | word |
| 22 | char_multi_distractor | head_mean_ablation | final | -0.2945824622780439 | symbolic |
| 20 | char_multi_distractor | head_mean_ablation | final | -0.2937088847117663 | symbolic |
| 20 | char_multi_distractor | head_zero_ablation | final | -0.2655849206080836 | symbolic |
| 20 | char_multi_distractor | head_zero_ablation | final | -0.2615743809027803 | symbolic |
| 22 | char_multi_distractor | head_zero_ablation | final | -0.2615743809027803 | symbolic |
| 20 | char_multi_distractor | head_clean_to_corrupt_patch | final | -0.226935571266719 | symbolic |

## Controls that moved

| seed | family | intervention | position_label | effect_size | token_domain |
| --- | --- | --- | --- | --- | --- |
| 20 | char_target_swap_control | head_clean_to_corrupt_patch | final | 1.0169673437820603 | symbolic |
| 20 | char_target_swap_control | head_zero_ablation | final | 0.5231623972094638 | symbolic |
| 22 | char_reversed_control | head_zero_ablation | final | 0.3674001126711735 | symbolic |
| 21 | char_reversed_control | head_clean_to_corrupt_patch | final | 0.1641904038420013 | symbolic |
| 22 | char_reversed_control | head_mean_ablation | final | 0.1317873548275264 | symbolic |
| 22 | char_target_swap_control | head_zero_ablation | final | 0.051981049236594 | symbolic |
| 21 | char_reversed_control | head_mean_ablation | final | 0.0517998974980683 | symbolic |
| 21 | char_reversed_control | head_zero_ablation | final | 0.0392720281594139 | symbolic |

## Negative-control comparison

Top negative-control rows by mean gap: `[{'layer': 4, 'head': 1, 'mean_positive_minus_control_gap': -0.0066721489855266, 'final_characterization_status': 'falsified_candidate'}, {'layer': 11, 'head': 0, 'mean_positive_minus_control_gap': -0.0096445696267726, 'final_characterization_status': 'falsified_candidate'}, {'layer': 11, 'head': 7, 'mean_positive_minus_control_gap': -0.0152257515614516, 'final_characterization_status': 'falsified_candidate'}, {'layer': 11, 'head': 9, 'mean_positive_minus_control_gap': -0.0172443501949977, 'final_characterization_status': 'falsified_candidate'}, {'layer': 9, 'head': 5, 'mean_positive_minus_control_gap': -0.0193007068378529, 'final_characterization_status': 'falsified_candidate'}]`.

## Token domains where it failed

| seed | family | intervention | position_label | effect_size | token_domain |
| --- | --- | --- | --- | --- | --- |
| 20 | char_word_short | head_zero_ablation | final | -0.3867558568378301 | word |
| 21 | char_word_short | head_zero_ablation | final | -0.3867558568378301 | word |
| 21 | char_word_long | head_mean_ablation | final | -0.1445188934583121 | word |
| 20 | char_word_long | head_zero_ablation | final | -0.1341932903354832 | word |
| 21 | char_word_long | head_zero_ablation | final | -0.1341932903354832 | word |
| 20 | char_word_long | head_mean_ablation | final | -0.1293450252860491 | word |
| 21 | char_number_short | head_mean_ablation | final | -0.1240863808129464 | number |
| 21 | char_word_short | head_zero_ablation | final | -0.1089581101004331 | word |

## Sequence lengths where it failed

| seed | family | intervention | position_label | effect_size | token_domain | sequence_length_bucket |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | char_word_short | head_zero_ablation | final | -0.3867558568378301 | word | short |
| 21 | char_word_short | head_zero_ablation | final | -0.3867558568378301 | word | short |
| 21 | char_word_long | head_mean_ablation | final | -0.1445188934583121 | word | long |
| 20 | char_word_long | head_zero_ablation | final | -0.1341932903354832 | word | long |
| 21 | char_word_long | head_zero_ablation | final | -0.1341932903354832 | word | long |
| 20 | char_word_long | head_mean_ablation | final | -0.1293450252860491 | word | long |
| 21 | char_number_short | head_mean_ablation | final | -0.1240863808129464 | number | short |
| 21 | char_word_short | head_zero_ablation | final | -0.1089581101004331 | word | short |

## Position/intervention mismatch

| seed | family | intervention | position_label | effect_size | token_domain |
| --- | --- | --- | --- | --- | --- |
| 20 | char_word_short | head_zero_ablation | final | -0.3867558568378301 | word |
| 21 | char_word_short | head_zero_ablation | final | -0.3867558568378301 | word |
| 22 | char_multi_distractor | head_mean_ablation | final | -0.2945824622780439 | symbolic |
| 20 | char_multi_distractor | head_mean_ablation | final | -0.2937088847117663 | symbolic |
| 20 | char_multi_distractor | head_zero_ablation | final | -0.2655849206080836 | symbolic |
| 20 | char_multi_distractor | head_zero_ablation | final | -0.2615743809027803 | symbolic |
| 22 | char_multi_distractor | head_zero_ablation | final | -0.2615743809027803 | symbolic |
| 21 | char_multi_distractor | head_mean_ablation | previous_occurrence | -0.1943753344403193 | symbolic |

## Attention/effect mismatch

Read this alongside the attention/effect alignment artifact. A positive or negative single-example effect is not enough without candidate-level alignment.

## OV/QK mismatch

Read this alongside the OV/QK diagnostic artifact. OV and QK margins are local signatures only, and they did not establish a complete circuit.

## Interpretation

The counterexamples are part of the result. This candidate should not be upgraded unless its failures, moved controls, and position/intervention sensitivity are explained.

## What not to claim

Do not claim induction-head discovery, a circuit, or broad GPT-2 behavior from this artifact.
