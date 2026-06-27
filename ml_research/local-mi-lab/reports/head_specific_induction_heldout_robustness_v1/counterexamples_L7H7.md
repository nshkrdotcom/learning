# Held-Out Counterexamples: L7H7

## Why this head was inspected

This head was part of the fixed held-out candidate set after the earlier head-specific sweep. The goal is to inspect where the held-out causal result succeeds, fails, or moves controls.

## Strongest successes

| seed | family | example_id | intervention | position_label | effect_size | true_expected_next_token | wrong_or_control_token | clean_prompt | corrupt_prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | heldout_symbolic_longer | heldout_symbolic_longer_0003 | head_zero_ablation | final | 2.8316 |  M |  N | N O P Q R M N O P Q R | N P R O Q M P N R O Q |
| 11 | heldout_symbolic_longer | heldout_symbolic_longer_0000 | head_zero_ablation | final | 2.8316 |  M |  N | N O P Q R M N O P Q R | N P R O Q M P N R O Q |
| 12 | heldout_symbolic_longer | heldout_symbolic_longer_0017 | head_zero_ablation | final | 1.2135 |  N |  O | O P Q R M N O P Q R M | O Q M P R N Q O M P R |
| 10 | heldout_symbolic_longer | heldout_symbolic_longer_0005 | head_zero_ablation | final | 1.2135 |  N |  O | O P Q R M N O P Q R M | O Q M P R N Q O M P R |
| 11 | heldout_word_sequences | heldout_word_sequences_0009 | head_zero_ablation | final | 0.7118 |  south |  east | east west north south east west north | east west north south north east west west |
| 10 | heldout_word_sequences | heldout_word_sequences_0015 | head_mean_ablation | final | 0.7079 |  spring |  summer | summer autumn winter spring summer autumn winter | summer autumn winter spring winter summer autumn autumn |
| 11 | heldout_word_sequences | heldout_word_sequences_0003 | head_mean_ablation | final | 0.4830 |  east |  west | west north south east west north south | west north south east south west north north |
| 10 | heldout_word_sequences | heldout_word_sequences_0021 | head_mean_ablation | final | 0.4587 |  east |  west | west north south east west north south | west north south east south west north north |
| 12 | heldout_symbolic_longer | heldout_symbolic_longer_0009 | head_mean_ablation | final | 0.4355 |  H |  I | I J K L G H I J K L G | I K G J L H K I G J L |
| 10 | heldout_word_sequences | heldout_word_sequences_0015 | head_clean_to_corrupt_patch | final | 0.4213 |  spring |  summer | summer autumn winter spring summer autumn winter | summer autumn winter spring winter summer autumn autumn |
| 12 | heldout_word_sequences | heldout_word_sequences_0000 | head_mean_ablation | final | 0.4151 |  east |  west | west north south east west north south | west north south east south west north north |
| 12 | heldout_symbolic_longer | heldout_symbolic_longer_0003 | head_zero_ablation | previous_occurrence | 0.4062 |  M |  N | N O P Q R M N O P Q R | N P R O Q M P N R O Q |

## Strongest failures

| seed | family | example_id | intervention | position_label | effect_size | true_expected_next_token | wrong_or_control_token | clean_prompt | corrupt_prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | heldout_word_sequences | heldout_word_sequences_0015 | head_zero_ablation | final | -1.2496 |  west |  north | north south east west north south east | north south east west east north south south |
| 12 | heldout_symbolic_longer | heldout_symbolic_longer_0009 | head_zero_ablation | final | -1.1222 |  H |  I | I J K L G H I J K L G | I K G J L H K I G J L |
| 11 | heldout_word_sequences | heldout_word_sequences_0015 | head_mean_ablation | final | -0.9613 |  west |  north | north south east west north south east | north south east west east north south south |
| 11 | heldout_symbolic_longer | heldout_symbolic_longer_0000 | head_clean_to_corrupt_patch | final | -0.7174 |  M |  N | N O P Q R M N O P Q R | N P R O Q M P N R O Q |
| 12 | heldout_symbolic_longer | heldout_symbolic_longer_0003 | head_clean_to_corrupt_patch | final | -0.7174 |  M |  N | N O P Q R M N O P Q R | N P R O Q M P N R O Q |
| 12 | heldout_word_sequences | heldout_word_sequences_0014 | head_zero_ablation | final | -0.6757 |  north |  south | south east west north south east west | south east west north west south east east |
| 11 | heldout_symbolic_longer | heldout_symbolic_longer_0000 | head_mean_ablation | final | -0.6229 |  M |  N | N O P Q R M N O P Q R | N P R O Q M P N R O Q |
| 10 | heldout_number_sequences | heldout_number_sequences_0022 | head_zero_ablation | final | -0.3592 |  six |  eight | eight two four six eight two four | eight two four six four eight two two |
| 11 | heldout_number_sequences | heldout_number_sequences_0018 | head_zero_ablation | final | -0.3592 |  six |  eight | eight two four six eight two four | eight two four six four eight two two |
| 10 | heldout_number_sequences | heldout_number_sequences_0006 | head_zero_ablation | final | -0.3592 |  six |  eight | eight two four six eight two four | eight two four six four eight two two |
| 12 | heldout_number_sequences | heldout_number_sequences_0000 | head_zero_ablation | final | -0.3592 |  six |  eight | eight two four six eight two four | eight two four six four eight two two |
| 11 | heldout_symbolic_longer | heldout_symbolic_longer_0020 | head_zero_ablation | final | -0.3533 |  T |  U | U V W X S T U V W X S | U W S V X T W U S V X |

## Controls that moved

| seed | family | example_id | intervention | position_label | effect_size | true_expected_next_token | wrong_or_control_token | clean_prompt | corrupt_prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0010 | head_zero_ablation | final | 2.8316 |  M |  N | N O P Q R M N O P Q R | N P R O Q M P N R O Q |
| 10 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0005 | head_zero_ablation | final | 1.2135 |  N |  O | O P Q R M N O P Q R M | O Q M P R N Q O M P R |
| 11 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0016 | head_zero_ablation | final | 1.2135 |  N |  O | O P Q R M N O P Q R M | O Q M P R N Q O M P R |
| 10 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0014 | head_zero_ablation | final | 0.6583 |  B |  C | C D E F A B C D E F A | C E A D F B E C A D F |
| 12 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0000 | head_zero_ablation | final | 0.6583 |  B |  C | C D E F A B C D E F A | C E A D F B E C A D F |
| 10 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0004 | head_mean_ablation | final | 0.5625 |  H |  I | I J K L G H I J K L G | I K G J L H K I G J L |
| 12 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0009 | head_mean_ablation | final | 0.4355 |  H |  I | I J K L G H I J K L G | I K G J L H K I G J L |
| 10 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0014 | head_clean_to_corrupt_patch | final | 0.3136 |  B |  C | C D E F A B C D E F A | C E A D F B E C A D F |
| 12 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0000 | head_clean_to_corrupt_patch | final | 0.3136 |  B |  C | C D E F A B C D E F A | C E A D F B E C A D F |
| 11 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0016 | head_mean_ablation | final | 0.2598 |  N |  O | O P Q R M N O P Q R M | O Q M P R N Q O M P R |
| 12 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0006 | head_zero_ablation | final | 0.1591 |  G |  H | H I J K L G H I J K L | H J L I K G J H L I K |
| 10 | heldout_no_structure_same_tokens | heldout_no_structure_same_tokens_0006 | head_zero_ablation | final | 0.1591 |  G |  H | H I J K L G H I J K L | H J L I K G J H L I K |

## Prompt constructions that broke it

- heldout_double_repeat: positive-family mean effect `-0.0019` over 216 valid rows.
- heldout_number_sequences: positive-family mean effect `-0.0017` over 216 valid rows.
- heldout_no_structure_same_tokens: control-family mean effect `0.0156` over 108 valid rows.

## Intervention/position sensitivity

- head_clean_to_corrupt_patch at final: mean effect `-0.0208`, valid rows `180`, controls moved `12`.
- head_clean_to_corrupt_patch at previous_occurrence: mean effect `0.0034`, valid rows `144`, controls moved `0`.
- head_mean_ablation at final: mean effect `0.0020`, valid rows `180`, controls moved `21`.
- head_mean_ablation at previous_occurrence: mean effect `0.0050`, valid rows `144`, controls moved `0`.
- head_zero_ablation at final: mean effect `0.0433`, valid rows `180`, controls moved `11`.
- head_zero_ablation at previous_occurrence: mean effect `0.0062`, valid rows `144`, controls moved `0`.

## Wrong-target failures

No rows in this bucket.

## Interpretation

Inspection buckets: `{'strongest_positive_success': 446, 'strongest_positive_failure': 418, 'invalid_or_unavailable': 324, 'other': 64, 'controls_that_moved': 44}`.

A useful candidate should move positive examples more than controls across held-out families and intervention variants. Rows where controls move or positive effects vanish are counterexamples for this lab stage.

## What not to claim

Do not claim induction-head discovery, a circuit, or broad GPT-2 behavior from these examples. This is counterexample-oriented practice evidence.
