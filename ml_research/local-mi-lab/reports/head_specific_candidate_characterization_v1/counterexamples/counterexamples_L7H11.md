# Candidate Characterization Counterexamples: L7H11

## Why this candidate was inspected

L7H11 was in the fixed primary candidate set from the prior held-out pass. Its final characterization status was `falsified_candidate`.

## Strongest successes

| seed | family | intervention | position_label | effect_size | token_domain |
| --- | --- | --- | --- | --- | --- |
| 21 | char_multi_distractor | head_clean_to_corrupt_patch | final | 0.4336695160708884 | symbolic |
| 20 | char_multi_distractor | head_zero_ablation | final | 0.4334777412221003 | symbolic |
| 22 | char_multi_distractor | head_zero_ablation | final | 0.4334777412221003 | symbolic |
| 21 | char_word_short | head_zero_ablation | final | 0.3404045757191024 | word |
| 22 | char_word_short | head_zero_ablation | final | 0.3404045757191024 | word |
| 21 | char_multi_distractor | head_mean_ablation | final | 0.302595540675487 | symbolic |
| 22 | char_multi_distractor | head_mean_ablation | final | 0.2796254455867297 | symbolic |
| 21 | char_number_short | head_zero_ablation | final | 0.2755969546595004 | number |

## Strongest failures

| seed | family | intervention | position_label | effect_size | token_domain |
| --- | --- | --- | --- | --- | --- |
| 21 | char_multi_distractor | head_clean_to_corrupt_patch | final | -1.368885603482041 | symbolic |
| 21 | char_multi_distractor | head_mean_ablation | final | -1.2629900757421126 | symbolic |
| 21 | char_multi_distractor | head_zero_ablation | final | -0.9231049301330038 | symbolic |
| 21 | char_multi_distractor | head_zero_ablation | previous_occurrence | -0.9210307105856912 | symbolic |
| 21 | char_multi_distractor | head_mean_ablation | previous_occurrence | -0.9070186955586326 | symbolic |
| 20 | char_multi_distractor | head_zero_ablation | final | -0.7294111753219035 | symbolic |
| 22 | char_multi_distractor | head_zero_ablation | final | -0.7294111753219035 | symbolic |
| 21 | char_multi_distractor | head_clean_to_corrupt_patch | previous_occurrence | -0.660181052887082 | symbolic |

## Controls that moved

| seed | family | intervention | position_label | effect_size | token_domain |
| --- | --- | --- | --- | --- | --- |
| 20 | char_target_swap_control | head_zero_ablation | final | 2.8701169562952797 | symbolic |
| 22 | char_reversed_control | head_clean_to_corrupt_patch | final | 0.6611615959499338 | symbolic |
| 20 | char_target_swap_control | head_clean_to_corrupt_patch | final | 0.5886642359960226 | symbolic |
| 22 | char_reversed_control | head_mean_ablation | final | 0.4029633678539009 | symbolic |
| 21 | char_reversed_control | head_zero_ablation | final | 0.3217754167067742 | symbolic |
| 21 | char_reversed_control | head_zero_ablation | final | 0.1290515299861094 | symbolic |
| 22 | char_target_swap_control | head_zero_ablation | final | 0.1176253528281361 | symbolic |
| 21 | char_reversed_control | head_clean_to_corrupt_patch | final | 0.1168404950270953 | symbolic |

## Negative-control comparison

Top negative-control rows by mean gap: `[{'layer': 4, 'head': 1, 'mean_positive_minus_control_gap': -0.0066721489855266, 'final_characterization_status': 'falsified_candidate'}, {'layer': 11, 'head': 0, 'mean_positive_minus_control_gap': -0.0096445696267726, 'final_characterization_status': 'falsified_candidate'}, {'layer': 11, 'head': 7, 'mean_positive_minus_control_gap': -0.0152257515614516, 'final_characterization_status': 'falsified_candidate'}, {'layer': 11, 'head': 9, 'mean_positive_minus_control_gap': -0.0172443501949977, 'final_characterization_status': 'falsified_candidate'}, {'layer': 9, 'head': 5, 'mean_positive_minus_control_gap': -0.0193007068378529, 'final_characterization_status': 'falsified_candidate'}]`.

## Token domains where it failed

| seed | family | intervention | position_label | effect_size | token_domain |
| --- | --- | --- | --- | --- | --- |
| 21 | char_multi_distractor | head_clean_to_corrupt_patch | final | -1.368885603482041 | symbolic |
| 21 | char_multi_distractor | head_mean_ablation | final | -1.2629900757421126 | symbolic |
| 21 | char_multi_distractor | head_zero_ablation | final | -0.9231049301330038 | symbolic |
| 21 | char_multi_distractor | head_zero_ablation | previous_occurrence | -0.9210307105856912 | symbolic |
| 21 | char_multi_distractor | head_mean_ablation | previous_occurrence | -0.9070186955586326 | symbolic |
| 20 | char_multi_distractor | head_zero_ablation | final | -0.7294111753219035 | symbolic |
| 22 | char_multi_distractor | head_zero_ablation | final | -0.7294111753219035 | symbolic |
| 21 | char_multi_distractor | head_clean_to_corrupt_patch | previous_occurrence | -0.660181052887082 | symbolic |

## Sequence lengths where it failed

| seed | family | intervention | position_label | effect_size | token_domain | sequence_length_bucket |
| --- | --- | --- | --- | --- | --- | --- |
| 21 | char_multi_distractor | head_clean_to_corrupt_patch | final | -1.368885603482041 | symbolic | medium |
| 21 | char_multi_distractor | head_mean_ablation | final | -1.2629900757421126 | symbolic | medium |
| 21 | char_multi_distractor | head_zero_ablation | final | -0.9231049301330038 | symbolic | medium |
| 21 | char_multi_distractor | head_zero_ablation | previous_occurrence | -0.9210307105856912 | symbolic | medium |
| 21 | char_multi_distractor | head_mean_ablation | previous_occurrence | -0.9070186955586326 | symbolic | medium |
| 20 | char_multi_distractor | head_zero_ablation | final | -0.7294111753219035 | symbolic | medium |
| 22 | char_multi_distractor | head_zero_ablation | final | -0.7294111753219035 | symbolic | medium |
| 21 | char_multi_distractor | head_clean_to_corrupt_patch | previous_occurrence | -0.660181052887082 | symbolic | medium |

## Position/intervention mismatch

| seed | family | intervention | position_label | effect_size | token_domain |
| --- | --- | --- | --- | --- | --- |
| 21 | char_multi_distractor | head_mean_ablation | final | -1.2629900757421126 | symbolic |
| 21 | char_multi_distractor | head_zero_ablation | previous_occurrence | -0.9210307105856912 | symbolic |
| 21 | char_multi_distractor | head_mean_ablation | previous_occurrence | -0.9070186955586326 | symbolic |
| 21 | char_multi_distractor | head_clean_to_corrupt_patch | previous_occurrence | -0.660181052887082 | symbolic |
| 22 | char_multi_distractor | head_mean_ablation | final | -0.3039715946391352 | symbolic |
| 20 | char_multi_distractor | head_mean_ablation | final | -0.3029226753391245 | symbolic |
| 20 | char_number_short | head_mean_ablation | final | -0.1422640373697686 | number |
| 22 | char_number_short | head_mean_ablation | final | -0.1255971906042223 | number |

## Attention/effect mismatch

Read this alongside the attention/effect alignment artifact. A positive or negative single-example effect is not enough without candidate-level alignment.

## OV/QK mismatch

Read this alongside the OV/QK diagnostic artifact. OV and QK margins are local signatures only, and they did not establish a complete circuit.

## Interpretation

The counterexamples are part of the result. This candidate should not be upgraded unless its failures, moved controls, and position/intervention sensitivity are explained.

## What not to claim

Do not claim induction-head discovery, a circuit, or broad GPT-2 behavior from this artifact.
