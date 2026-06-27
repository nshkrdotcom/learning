# Head-Specific Induction Held-Out Robustness v1 Result

## Runs

- Seed 10 held-out robustness run: `runs/20260626_161445_gpt2_small_induction_heldout_seed10`
- Seed 11 held-out robustness run: `runs/20260626_162154_gpt2_small_induction_heldout_seed11`
- Seed 12 held-out robustness run: `runs/20260626_162753_gpt2_small_induction_heldout_seed12`

An earlier seed 10 attempt, `runs/20260626_160547_gpt2_small_induction_heldout_seed10`, is intentionally not used because it had invalid clean/corrupt construction for positives and produced denominator-zero artifacts.

## Candidate Set

The fixed candidate set is stored at:

- `reports/head_specific_induction_heldout_robustness_v1/heldout_candidate_set.csv`

The primary candidates were:

- L7H7
- L9H11
- L7H11
- L7H0
- L0H8

The prior raw-attention comparison heads were:

- L0H1
- L0H5
- L0H10
- L11H8
- L0H4

The negative-control heads were carried forward from the held-out candidate set:

- L9H5
- L11H0
- L4H1
- L4H9
- L11H9
- L11H7

## Held-Out Prompt Families

- `heldout_symbolic_longer`
- `heldout_word_sequences`
- `heldout_number_sequences`
- `heldout_double_repeat`
- `heldout_wrong_target_same_prompt`
- `heldout_no_structure_same_tokens`

## Interventions and Positions

Interventions:

- `head_clean_to_corrupt_patch`
- `head_zero_ablation`
- `head_mean_ablation`

Positions:

- `final`
- `previous_occurrence`

Previous-occurrence rows were often unavailable or insufficient for controls without meaningful source-position metadata. Those rows are recorded as insufficient rather than filled in.

## Consolidated Statuses

The consolidated report is stored at:

- `reports/head_specific_induction_heldout_robustness_v1/head_specific_induction_heldout_robustness_v1.md`
- `reports/head_specific_induction_heldout_robustness_v1/heldout_multiseed_summary.json`
- `reports/head_specific_induction_heldout_robustness_v1/heldout_multiseed_by_candidate.csv`

Consolidated status counts:

- `heldout_falsified`: 11
- `heldout_downgraded`: 5
- `heldout_replicated`: 0

No candidate cleanly survived the held-out robustness rule.

## L7H7

Final status: `heldout_falsified`.

Details:

- Candidate group: `random_comparison_replicated`
- Mean positive-minus-control gap: `-0.0094`
- Minimum positive-minus-control gap: `-0.2248`
- Survived seed-level rows: 3
- Survived interventions: `head_clean_to_corrupt_patch`, `head_zero_ablation`
- Survived positions: `final`

Interpretation: L7H7 had strong individual positive examples, but it also moved no-structure controls and remained final-position/intervention-sensitive. It is not a robust induction-head candidate.

## L9H11

Final status: `heldout_falsified`.

Details:

- Candidate group: `replicated_candidate`
- Mean positive-minus-control gap: `-0.0494`
- Minimum positive-minus-control gap: `-0.3557`
- Survived seed-level rows: 3
- Survived interventions: `head_clean_to_corrupt_patch`, `head_mean_ablation`, `head_zero_ablation`
- Survived positions: `final`

Interpretation: L9H11 had some seed-level survival rows, but aggregate held-out results included controls-moving and negative gaps. It is falsified for this lab stage.

## L7H11

Final status: `heldout_downgraded`.

Details:

- Candidate group: `replicated_candidate`
- Mean positive-minus-control gap: `0.1829`
- Minimum positive-minus-control gap: `-0.3509`
- Survived seed-level rows: 1
- Survived interventions: `head_clean_to_corrupt_patch`
- Survived positions: `final`

Interpretation: L7H11 had a strong seed-10 clean-to-corrupt result but did not replicate across held-out seeds and variants.

## L7H0

Final status: `heldout_falsified`.

Details:

- Candidate group: `replicated_candidate`
- Mean positive-minus-control gap: `0.0744`
- Minimum positive-minus-control gap: `-0.0960`
- Survived seed-level rows: 0

Interpretation: L7H0 did not produce useful held-out positive effects.

## L0H8

Final status: `heldout_downgraded`.

Details:

- Candidate group: `replicated_candidate`
- Mean positive-minus-control gap: `0.0006`
- Minimum positive-minus-control gap: `-0.1806`
- Survived seed-level rows: 2
- Survived interventions: `head_clean_to_corrupt_patch`, `head_mean_ablation`
- Survived positions: `final`

Interpretation: L0H8 had weak final-position support but a near-zero aggregate gap and no robust position-variant support.

## Prior Raw-Attention Heads

- L0H1: `heldout_downgraded`, mean gap `0.1430`, final-only support.
- L0H5: `heldout_downgraded`, mean gap `0.0595`, final-only support.
- L0H10: `heldout_falsified`, mean gap `0.1222`, controls/failures prevent robust interpretation.
- L0H4: `heldout_falsified`, mean gap `-0.0144`.
- L11H8: `heldout_falsified`, mean gap `-0.1685`.

Prior raw-attention heads still do not provide robust induction-head evidence.

## Negative Controls

Negative controls also produced apparent seed-level survival rows:

- L11H0: `heldout_falsified`, survived 3 seed-level rows, mean gap `0.1124`.
- L11H9: `heldout_downgraded`, survived 1 seed-level row, mean gap `-0.0038`.
- L4H1: `heldout_falsified`.
- L9H5: `heldout_falsified`.
- L4H9: `heldout_falsified`.
- L11H7: `heldout_falsified`.

This is important: negative controls passing permissive seed-level rules means the rule is not specific enough by itself.

## Counterexamples

Counterexample artifacts:

- `reports/head_specific_induction_heldout_robustness_v1/counterexamples_L7H7.md`
- `reports/head_specific_induction_heldout_robustness_v1/counterexamples_L9H11.md`
- `reports/head_specific_induction_heldout_robustness_v1/counterexamples_L7H11.md`

Inspection bucket counts:

- L7H7: `strongest_positive_success`: 446; `strongest_positive_failure`: 418; `invalid_or_unavailable`: 324; `other`: 64; `controls_that_moved`: 44.
- L9H11: `strongest_positive_success`: 487; `strongest_positive_failure`: 377; `invalid_or_unavailable`: 324; `controls_that_moved`: 72; `other`: 36.
- L7H11: `strongest_positive_success`: 478; `strongest_positive_failure`: 386; `invalid_or_unavailable`: 324; `other`: 57; `controls_that_moved`: 51.

The counterexamples show positive successes, but also substantial failures, controls moving, and intervention/position sensitivity.

## Final Interpretation

The held-out robustness check falsified or downgraded the previously replicated candidates. They should not be treated as induction-head candidates beyond the original synthetic setup.

This is useful progress: the lab pipeline is now strong enough to falsify its own earlier candidates.

## What This Does Not Show

This does not show an induction-head discovery.

This does not show a circuit.

This does not show broad GPT-2 behavior.

This does not show that raw attention is causal evidence.

This does not show that final-position patching alone is enough.

## Exact Reproduction Commands

```bash
uv run python scripts/compare_heldout_robustness_runs.py \
  --runs \
    runs/20260626_161445_gpt2_small_induction_heldout_seed10 \
    runs/20260626_162154_gpt2_small_induction_heldout_seed11 \
    runs/20260626_162753_gpt2_small_induction_heldout_seed12 \
  --output reports/head_specific_induction_heldout_robustness_v1

uv run python scripts/inspect_heldout_counterexamples.py \
  --candidate L7H7 \
  --runs \
    runs/20260626_161445_gpt2_small_induction_heldout_seed10 \
    runs/20260626_162154_gpt2_small_induction_heldout_seed11 \
    runs/20260626_162753_gpt2_small_induction_heldout_seed12 \
  --output reports/head_specific_induction_heldout_robustness_v1

uv run python scripts/inspect_heldout_counterexamples.py \
  --candidate L9H11 \
  --runs \
    runs/20260626_161445_gpt2_small_induction_heldout_seed10 \
    runs/20260626_162154_gpt2_small_induction_heldout_seed11 \
    runs/20260626_162753_gpt2_small_induction_heldout_seed12 \
  --output reports/head_specific_induction_heldout_robustness_v1

uv run python scripts/inspect_heldout_counterexamples.py \
  --candidate L7H11 \
  --runs \
    runs/20260626_161445_gpt2_small_induction_heldout_seed10 \
    runs/20260626_162154_gpt2_small_induction_heldout_seed11 \
    runs/20260626_162753_gpt2_small_induction_heldout_seed12 \
  --output reports/head_specific_induction_heldout_robustness_v1
```
