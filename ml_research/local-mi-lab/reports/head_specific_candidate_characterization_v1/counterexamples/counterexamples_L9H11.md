# Candidate Characterization Counterexamples: L9H11

## Why this candidate was inspected

L9H11 was in the fixed primary candidate set from the prior held-out pass. Its final characterization status was `falsified_candidate`.

## Strongest successes

| seed | family | intervention | position_label | effect_size | token_domain |
| --- | --- | --- | --- | --- | --- |
| 21 | char_multi_distractor | head_zero_ablation | final | 0.3926729013785702 | symbolic |
| 21 | char_multi_distractor | head_mean_ablation | final | 0.2624815092588372 | symbolic |
| 20 | char_multi_distractor | head_zero_ablation | final | 0.1265612459282622 | symbolic |
| 22 | char_multi_distractor | head_zero_ablation | final | 0.1265612459282622 | symbolic |
| 20 | char_word_short | head_zero_ablation | final | 0.1148976243721816 | word |
| 22 | char_word_short | head_zero_ablation | final | 0.1148976243721816 | word |
| 22 | char_multi_distractor | head_mean_ablation | final | 0.1082516028634975 | symbolic |
| 20 | char_multi_distractor | head_mean_ablation | final | 0.1059043197732246 | symbolic |

## Strongest failures

| seed | family | intervention | position_label | effect_size | token_domain |
| --- | --- | --- | --- | --- | --- |
| 21 | char_multi_distractor | head_zero_ablation | final | -0.6541625062750658 | symbolic |
| 21 | char_multi_distractor | head_mean_ablation | final | -0.5471085048517981 | symbolic |
| 20 | char_multi_distractor | head_zero_ablation | final | -0.1235224215138326 | symbolic |
| 21 | char_word_short | head_zero_ablation | final | -0.060844128483716 | word |
| 21 | char_multi_distractor | head_clean_to_corrupt_patch | final | -0.0575709485076884 | symbolic |
| 21 | char_word_short | head_mean_ablation | final | -0.0559090743879307 | word |
| 20 | char_multi_distractor | head_clean_to_corrupt_patch | final | -0.0477510856910985 | symbolic |
| 20 | char_word_short | head_zero_ablation | final | -0.0452719752736177 | word |

## Controls that moved

| seed | family | intervention | position_label | effect_size | token_domain |
| --- | --- | --- | --- | --- | --- |
| 22 | char_reversed_control | head_zero_ablation | final | 0.4538429753410573 | symbolic |
| 22 | char_reversed_control | head_clean_to_corrupt_patch | final | 0.2058089558260371 | symbolic |
| 22 | char_reversed_control | head_mean_ablation | final | 0.1351410267672676 | symbolic |
| 22 | char_target_swap_control | head_clean_to_corrupt_patch | final | 0.0591409443573855 | symbolic |
| 22 | char_target_swap_control | head_clean_to_corrupt_patch | final | 0.0380117056688691 | symbolic |
| 22 | char_target_swap_control | head_mean_ablation | final | 0.0364041960789951 | symbolic |
| 21 | char_reversed_control | head_zero_ablation | final | 0.0363638563676746 | symbolic |
| 21 | char_target_swap_control | head_mean_ablation | final | 0.0203601325652732 | symbolic |

## Negative-control comparison

Top negative-control rows by mean gap: `[{'layer': 4, 'head': 1, 'mean_positive_minus_control_gap': -0.0066721489855266, 'final_characterization_status': 'falsified_candidate'}, {'layer': 11, 'head': 0, 'mean_positive_minus_control_gap': -0.0096445696267726, 'final_characterization_status': 'falsified_candidate'}, {'layer': 11, 'head': 7, 'mean_positive_minus_control_gap': -0.0152257515614516, 'final_characterization_status': 'falsified_candidate'}, {'layer': 11, 'head': 9, 'mean_positive_minus_control_gap': -0.0172443501949977, 'final_characterization_status': 'falsified_candidate'}, {'layer': 9, 'head': 5, 'mean_positive_minus_control_gap': -0.0193007068378529, 'final_characterization_status': 'falsified_candidate'}]`.

## Token domains where it failed

| seed | family | intervention | position_label | effect_size | token_domain |
| --- | --- | --- | --- | --- | --- |
| 21 | char_multi_distractor | head_zero_ablation | final | -0.6541625062750658 | symbolic |
| 21 | char_multi_distractor | head_mean_ablation | final | -0.5471085048517981 | symbolic |
| 20 | char_multi_distractor | head_zero_ablation | final | -0.1235224215138326 | symbolic |
| 21 | char_multi_distractor | head_clean_to_corrupt_patch | final | -0.0575709485076884 | symbolic |
| 20 | char_multi_distractor | head_clean_to_corrupt_patch | final | -0.0477510856910985 | symbolic |
| 21 | char_multi_distractor | head_zero_ablation | final | -0.0387552594761083 | symbolic |
| 21 | char_multi_distractor | head_mean_ablation | previous_occurrence | -0.0376503947085335 | symbolic |
| 20 | char_number_short | head_zero_ablation | final | -0.0320788029019413 | number |

## Sequence lengths where it failed

| seed | family | intervention | position_label | effect_size | token_domain | sequence_length_bucket |
| --- | --- | --- | --- | --- | --- | --- |
| 21 | char_multi_distractor | head_zero_ablation | final | -0.6541625062750658 | symbolic | medium |
| 21 | char_multi_distractor | head_mean_ablation | final | -0.5471085048517981 | symbolic | medium |
| 20 | char_multi_distractor | head_zero_ablation | final | -0.1235224215138326 | symbolic | medium |
| 21 | char_multi_distractor | head_clean_to_corrupt_patch | final | -0.0575709485076884 | symbolic | medium |
| 20 | char_multi_distractor | head_clean_to_corrupt_patch | final | -0.0477510856910985 | symbolic | medium |
| 21 | char_multi_distractor | head_zero_ablation | final | -0.0387552594761083 | symbolic | medium |
| 21 | char_multi_distractor | head_mean_ablation | previous_occurrence | -0.0376503947085335 | symbolic | medium |
| 20 | char_number_short | head_zero_ablation | final | -0.0320788029019413 | number | short |

## Position/intervention mismatch

| seed | family | intervention | position_label | effect_size | token_domain |
| --- | --- | --- | --- | --- | --- |
| 21 | char_multi_distractor | head_mean_ablation | final | -0.5471085048517981 | symbolic |
| 21 | char_word_short | head_mean_ablation | final | -0.0559090743879307 | word |
| 21 | char_multi_distractor | head_mean_ablation | previous_occurrence | -0.0376503947085335 | symbolic |
| 22 | char_word_short | head_mean_ablation | final | -0.0342283643229692 | word |
| 21 | char_word_short | head_mean_ablation | final | -0.0308183895002463 | word |
| 21 | char_multi_distractor | head_clean_to_corrupt_patch | previous_occurrence | -0.0262918360704572 | symbolic |
| 21 | char_word_short | head_mean_ablation | final | -0.0241736645755546 | word |
| 21 | char_multi_distractor | head_mean_ablation | final | -0.0227886925466079 | symbolic |

## Attention/effect mismatch

Read this alongside the attention/effect alignment artifact. A positive or negative single-example effect is not enough without candidate-level alignment.

## OV/QK mismatch

Read this alongside the OV/QK diagnostic artifact. OV and QK margins are local signatures only, and they did not establish a complete circuit.

## Interpretation

The counterexamples are part of the result. This candidate should not be upgraded unless its failures, moved controls, and position/intervention sensitivity are explained.

## What not to claim

Do not claim induction-head discovery, a circuit, or broad GPT-2 behavior from this artifact.
