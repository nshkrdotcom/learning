# Held-Out Counterexamples: L7H11

## Why this head was inspected

This head was part of the fixed held-out candidate set after the earlier head-specific sweep. The goal is to inspect where the held-out causal result succeeds, fails, or moves controls.

## Strongest successes

| seed | family | example_id | intervention | position_label | effect_size | true_expected_next_token | wrong_or_control_token | clean_prompt | corrupt_prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | heldout_symbolic_longer | heldout_symbolic_longer_0017 | head_clean_to_corrupt_patch | final | 2.0315 |  N |  O | O P Q R M N O P Q R M | O Q M P R N Q O M P R |
| 10 | heldout_symbolic_longer | heldout_symbolic_longer_0005 | head_clean_to_corrupt_patch | final | 2.0315 |  N |  O | O P Q R M N O P Q R M | O Q M P R N Q O M P R |
| 11 | heldout_word_sequences | heldout_word_sequences_0015 | head_mean_ablation | final | 1.0511 |  west |  north | north south east west north south east | north south east west east north south south |
| 11 | heldout_symbolic_longer | heldout_symbolic_longer_0000 | head_zero_ablation | previous_occurrence | 1.0114 |  M |  N | N O P Q R M N O P Q R | N P R O Q M P N R O Q |
| 12 | heldout_symbolic_longer | heldout_symbolic_longer_0003 | head_zero_ablation | previous_occurrence | 1.0114 |  M |  N | N O P Q R M N O P Q R | N P R O Q M P N R O Q |
| 11 | heldout_symbolic_longer | heldout_symbolic_longer_0020 | head_mean_ablation | final | 0.7509 |  T |  U | U V W X S T U V W X S | U W S V X T W U S V X |
| 12 | heldout_symbolic_longer | heldout_symbolic_longer_0022 | head_mean_ablation | final | 0.7248 |  T |  U | U V W X S T U V W X S | U W S V X T W U S V X |
| 11 | heldout_word_sequences | heldout_word_sequences_0009 | head_mean_ablation | final | 0.6484 |  south |  east | east west north south east west north | east west north south north east west west |
| 12 | heldout_symbolic_longer | heldout_symbolic_longer_0022 | head_clean_to_corrupt_patch | final | 0.4806 |  T |  U | U V W X S T U V W X S | U W S V X T W U S V X |
| 11 | heldout_symbolic_longer | heldout_symbolic_longer_0020 | head_clean_to_corrupt_patch | final | 0.4806 |  T |  U | U V W X S T U V W X S | U W S V X T W U S V X |
| 12 | heldout_word_sequences | heldout_word_sequences_0011 | head_zero_ablation | previous_occurrence | 0.4044 |  winter |  spring | spring summer autumn winter spring summer autumn | spring summer autumn winter autumn spring summer summer |
| 11 | heldout_word_sequences | heldout_word_sequences_0010 | head_zero_ablation | previous_occurrence | 0.4044 |  winter |  spring | spring summer autumn winter spring summer autumn | spring summer autumn winter autumn spring summer summer |

## Strongest failures

| seed | family | example_id | intervention | position_label | effect_size | true_expected_next_token | wrong_or_control_token | clean_prompt | corrupt_prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | heldout_symbolic_longer | heldout_symbolic_longer_0000 | head_mean_ablation | final | -7.2178 |  M |  N | N O P Q R M N O P Q R | N P R O Q M P N R O Q |
| 12 | heldout_symbolic_longer | heldout_symbolic_longer_0003 | head_mean_ablation | final | -7.1034 |  M |  N | N O P Q R M N O P Q R | N P R O Q M P N R O Q |
| 12 | heldout_symbolic_longer | heldout_symbolic_longer_0003 | head_clean_to_corrupt_patch | final | -7.0056 |  M |  N | N O P Q R M N O P Q R | N P R O Q M P N R O Q |
| 11 | heldout_symbolic_longer | heldout_symbolic_longer_0000 | head_clean_to_corrupt_patch | final | -7.0056 |  M |  N | N O P Q R M N O P Q R | N P R O Q M P N R O Q |
| 10 | heldout_symbolic_longer | heldout_symbolic_longer_0005 | head_zero_ablation | final | -4.1258 |  N |  O | O P Q R M N O P Q R M | O Q M P R N Q O M P R |
| 12 | heldout_symbolic_longer | heldout_symbolic_longer_0017 | head_zero_ablation | final | -4.1258 |  N |  O | O P Q R M N O P Q R M | O Q M P R N Q O M P R |
| 12 | heldout_symbolic_longer | heldout_symbolic_longer_0009 | head_zero_ablation | final | -2.8835 |  H |  I | I J K L G H I J K L G | I K G J L H K I G J L |
| 10 | heldout_symbolic_longer | heldout_symbolic_longer_0005 | head_mean_ablation | final | -0.8075 |  N |  O | O P Q R M N O P Q R M | O Q M P R N Q O M P R |
| 11 | heldout_word_sequences | heldout_word_sequences_0015 | head_zero_ablation | final | -0.7920 |  west |  north | north south east west north south east | north south east west east north south south |
| 12 | heldout_symbolic_longer | heldout_symbolic_longer_0017 | head_mean_ablation | final | -0.7311 |  N |  O | O P Q R M N O P Q R M | O Q M P R N Q O M P R |
| 12 | heldout_symbolic_longer | heldout_symbolic_longer_0022 | head_zero_ablation | final | -0.6908 |  T |  U | U V W X S T U V W X S | U W S V X T W U S V X |
| 11 | heldout_symbolic_longer | heldout_symbolic_longer_0020 | head_zero_ablation | final | -0.6908 |  T |  U | U V W X S T U V W X S | U W S V X T W U S V X |

## Controls that moved

| seed | family | example_id | intervention | position_label | effect_size | true_expected_next_token | wrong_or_control_token | clean_prompt | corrupt_prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0005 | head_clean_to_corrupt_patch | final | 2.0315 |  N |  O | O P Q R M N O P Q R M | O Q M P R N Q O M P R |
| 11 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0016 | head_clean_to_corrupt_patch | final | 2.0315 |  N |  O | O P Q R M N O P Q R M | O Q M P R N Q O M P R |
| 10 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0014 | head_zero_ablation | final | 1.3776 |  B |  C | C D E F A B C D E F A | C E A D F B E C A D F |
| 12 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0000 | head_zero_ablation | final | 1.3776 |  B |  C | C D E F A B C D E F A | C E A D F B E C A D F |
| 11 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0020 | head_mean_ablation | final | 0.7509 |  T |  U | U V W X S T U V W X S | U W S V X T W U S V X |
| 10 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0018 | head_mean_ablation | final | 0.6862 |  T |  U | U V W X S T U V W X S | U W S V X T W U S V X |
| 10 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0018 | head_clean_to_corrupt_patch | final | 0.4806 |  T |  U | U V W X S T U V W X S | U W S V X T W U S V X |
| 11 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0020 | head_clean_to_corrupt_patch | final | 0.4806 |  T |  U | U V W X S T U V W X S | U W S V X T W U S V X |
| 11 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0010 | head_clean_to_corrupt_patch | final | 0.2504 |  O |  P | P Q R M N O P Q R M N | P R N Q M O R P N Q M |
| 12 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0001 | head_clean_to_corrupt_patch | final | 0.2504 |  O |  P | P Q R M N O P Q R M N | P R N Q M O R P N Q M |
| 11 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0011 | head_clean_to_corrupt_patch | final | 0.2205 |  I |  J | J K L G H I J K L G H | J L H K G I L J H K G |
| 10 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0003 | head_clean_to_corrupt_patch | final | 0.2205 |  I |  J | J K L G H I J K L G H | J L H K G I L J H K G |

## Prompt constructions that broke it

- heldout_symbolic_longer: positive-family mean effect `-0.1729` over 216 valid rows.
- heldout_double_repeat: positive-family mean effect `-0.0012` over 216 valid rows.

## Intervention/position sensitivity

- head_clean_to_corrupt_patch at final: mean effect `-0.0463`, valid rows `180`, controls moved `21`.
- head_clean_to_corrupt_patch at previous_occurrence: mean effect `-0.0010`, valid rows `144`, controls moved `0`.
- head_mean_ablation at final: mean effect `-0.0994`, valid rows `180`, controls moved `21`.
- head_mean_ablation at previous_occurrence: mean effect `0.0033`, valid rows `144`, controls moved `0`.
- head_zero_ablation at final: mean effect `-0.1838`, valid rows `180`, controls moved `9`.
- head_zero_ablation at previous_occurrence: mean effect `0.0313`, valid rows `144`, controls moved `0`.

## Wrong-target failures

No rows in this bucket.

## Interpretation

Inspection buckets: `{'strongest_positive_success': 478, 'strongest_positive_failure': 386, 'invalid_or_unavailable': 324, 'other': 57, 'controls_that_moved': 51}`.

A useful candidate should move positive examples more than controls across held-out families and intervention variants. Rows where controls move or positive effects vanish are counterexamples for this lab stage.

## What not to claim

Do not claim induction-head discovery, a circuit, or broad GPT-2 behavior from these examples. This is counterexample-oriented practice evidence.
