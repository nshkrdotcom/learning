# Held-Out Counterexamples: L9H11

## Why this head was inspected

This head was part of the fixed held-out candidate set after the earlier head-specific sweep. The goal is to inspect where the held-out causal result succeeds, fails, or moves controls.

## Strongest successes

| seed | family | example_id | intervention | position_label | effect_size | true_expected_next_token | wrong_or_control_token | clean_prompt | corrupt_prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | heldout_symbolic_longer | heldout_symbolic_longer_0017 | head_zero_ablation | final | 3.7625 |  N |  O | O P Q R M N O P Q R M | O Q M P R N Q O M P R |
| 10 | heldout_symbolic_longer | heldout_symbolic_longer_0005 | head_zero_ablation | final | 3.7625 |  N |  O | O P Q R M N O P Q R M | O Q M P R N Q O M P R |
| 12 | heldout_symbolic_longer | heldout_symbolic_longer_0003 | head_zero_ablation | final | 1.1969 |  M |  N | N O P Q R M N O P Q R | N P R O Q M P N R O Q |
| 11 | heldout_symbolic_longer | heldout_symbolic_longer_0000 | head_zero_ablation | final | 1.1969 |  M |  N | N O P Q R M N O P Q R | N P R O Q M P N R O Q |
| 12 | heldout_symbolic_longer | heldout_symbolic_longer_0014 | head_zero_ablation | final | 0.5728 |  Q |  R | R M N O P Q R M N O P | R N P M O Q N R P M O |
| 12 | heldout_symbolic_longer | heldout_symbolic_longer_0017 | head_clean_to_corrupt_patch | final | 0.3860 |  N |  O | O P Q R M N O P Q R M | O Q M P R N Q O M P R |
| 10 | heldout_symbolic_longer | heldout_symbolic_longer_0005 | head_clean_to_corrupt_patch | final | 0.3860 |  N |  O | O P Q R M N O P Q R M | O Q M P R N Q O M P R |
| 12 | heldout_symbolic_longer | heldout_symbolic_longer_0009 | head_zero_ablation | final | 0.3502 |  H |  I | I J K L G H I J K L G | I K G J L H K I G J L |
| 11 | heldout_symbolic_longer | heldout_symbolic_longer_0020 | head_zero_ablation | final | 0.2995 |  T |  U | U V W X S T U V W X S | U W S V X T W U S V X |
| 12 | heldout_symbolic_longer | heldout_symbolic_longer_0022 | head_zero_ablation | final | 0.2995 |  T |  U | U V W X S T U V W X S | U W S V X T W U S V X |
| 11 | heldout_symbolic_longer | heldout_symbolic_longer_0011 | head_zero_ablation | final | 0.2439 |  I |  J | J K L G H I J K L G H | J L H K G I L J H K G |
| 10 | heldout_symbolic_longer | heldout_symbolic_longer_0003 | head_zero_ablation | final | 0.2439 |  I |  J | J K L G H I J K L G H | J L H K G I L J H K G |

## Strongest failures

| seed | family | example_id | intervention | position_label | effect_size | true_expected_next_token | wrong_or_control_token | clean_prompt | corrupt_prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | heldout_symbolic_longer | heldout_symbolic_longer_0009 | head_zero_ablation | final | -0.3543 |  D |  E | E F A B C D E F A B C | E A C F B D A E C F B |
| 10 | heldout_word_sequences | heldout_word_sequences_0015 | head_zero_ablation | final | -0.2070 |  spring |  summer | summer autumn winter spring summer autumn winter | summer autumn winter spring winter summer autumn autumn |
| 10 | heldout_symbolic_longer | heldout_symbolic_longer_0009 | head_mean_ablation | final | -0.1536 |  D |  E | E F A B C D E F A B C | E A C F B D A E C F B |
| 10 | heldout_symbolic_longer | heldout_symbolic_longer_0009 | head_clean_to_corrupt_patch | final | -0.1400 |  D |  E | E F A B C D E F A B C | E A C F B D A E C F B |
| 11 | heldout_word_sequences | heldout_word_sequences_0015 | head_zero_ablation | final | -0.1383 |  west |  north | north south east west north south east | north south east west east north south south |
| 10 | heldout_word_sequences | heldout_word_sequences_0015 | head_mean_ablation | final | -0.1133 |  spring |  summer | summer autumn winter spring summer autumn winter | summer autumn winter spring winter summer autumn autumn |
| 12 | heldout_symbolic_longer | heldout_symbolic_longer_0009 | head_clean_to_corrupt_patch | final | -0.1039 |  H |  I | I J K L G H I J K L G | I K G J L H K I G J L |
| 11 | heldout_double_repeat | heldout_double_repeat_0011 | head_mean_ablation | final | -0.0975 |  blue |  green | green black white red blue green black white red | green black white red blue green white black |
| 12 | heldout_word_sequences | heldout_word_sequences_0017 | head_zero_ablation | final | -0.0884 |  autumn |  winter | winter spring summer autumn winter spring summer | winter spring summer autumn summer winter spring spring |
| 10 | heldout_word_sequences | heldout_word_sequences_0001 | head_zero_ablation | final | -0.0884 |  autumn |  winter | winter spring summer autumn winter spring summer | winter spring summer autumn summer winter spring spring |
| 10 | heldout_word_sequences | heldout_word_sequences_0001 | head_mean_ablation | final | -0.0716 |  autumn |  winter | winter spring summer autumn winter spring summer | winter spring summer autumn summer winter spring spring |
| 12 | heldout_word_sequences | heldout_word_sequences_0017 | head_mean_ablation | final | -0.0640 |  autumn |  winter | winter spring summer autumn winter spring summer | winter spring summer autumn summer winter spring spring |

## Controls that moved

| seed | family | example_id | intervention | position_label | effect_size | true_expected_next_token | wrong_or_control_token | clean_prompt | corrupt_prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0016 | head_zero_ablation | final | 3.7625 |  N |  O | O P Q R M N O P Q R M | O Q M P R N Q O M P R |
| 10 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0005 | head_zero_ablation | final | 3.7625 |  N |  O | O P Q R M N O P Q R M | O Q M P R N Q O M P R |
| 10 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0010 | head_zero_ablation | final | 1.1969 |  M |  N | N O P Q R M N O P Q R | N P R O Q M P N R O Q |
| 10 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0005 | head_clean_to_corrupt_patch | final | 0.3860 |  N |  O | O P Q R M N O P Q R M | O Q M P R N Q O M P R |
| 11 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0016 | head_clean_to_corrupt_patch | final | 0.3860 |  N |  O | O P Q R M N O P Q R M | O Q M P R N Q O M P R |
| 10 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0004 | head_zero_ablation | final | 0.3502 |  H |  I | I J K L G H I J K L G | I K G J L H K I G J L |
| 12 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0009 | head_zero_ablation | final | 0.3502 |  H |  I | I J K L G H I J K L G | I K G J L H K I G J L |
| 10 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0018 | head_zero_ablation | final | 0.2995 |  T |  U | U V W X S T U V W X S | U W S V X T W U S V X |
| 11 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0020 | head_zero_ablation | final | 0.2995 |  T |  U | U V W X S T U V W X S | U W S V X T W U S V X |
| 11 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0011 | head_zero_ablation | final | 0.2439 |  I |  J | J K L G H I J K L G H | J L H K G I L J H K G |
| 10 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0003 | head_zero_ablation | final | 0.2439 |  I |  J | J K L G H I J K L G H | J L H K G I L J H K G |
| 10 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0005 | head_mean_ablation | final | 0.1021 |  N |  O | O P Q R M N O P Q R M | O Q M P R N Q O M P R |

## Prompt constructions that broke it

- heldout_no_structure_same_tokens: control-family mean effect `0.0848` over 108 valid rows.

## Intervention/position sensitivity

- head_clean_to_corrupt_patch at final: mean effect `0.0087`, valid rows `180`, controls moved `25`.
- head_clean_to_corrupt_patch at previous_occurrence: mean effect `0.0007`, valid rows `144`, controls moved `0`.
- head_mean_ablation at final: mean effect `-0.0000`, valid rows `180`, controls moved `23`.
- head_mean_ablation at previous_occurrence: mean effect `0.0003`, valid rows `144`, controls moved `0`.
- head_zero_ablation at final: mean effect `0.1272`, valid rows `180`, controls moved `24`.
- head_zero_ablation at previous_occurrence: mean effect `0.0007`, valid rows `144`, controls moved `0`.

## Wrong-target failures

No rows in this bucket.

## Interpretation

Inspection buckets: `{'strongest_positive_success': 487, 'strongest_positive_failure': 377, 'invalid_or_unavailable': 324, 'controls_that_moved': 72, 'other': 36}`.

A useful candidate should move positive examples more than controls across held-out families and intervention variants. Rows where controls move or positive effects vanish are counterexamples for this lab stage.

## What not to claim

Do not claim induction-head discovery, a circuit, or broad GPT-2 behavior from these examples. This is counterexample-oriented practice evidence.
