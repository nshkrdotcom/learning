# Head-Specific Candidate Characterization Failure Taxonomy v1

## Source Artifacts

- Counterexamples: `reports/head_specific_candidate_characterization_v1/counterexamples`
- Characterization summary: `docs/results/head_specific_candidate_characterization_v1.md`

## Command

```bash
uv run python scripts/build_characterization_failure_taxonomy.py --counterexamples reports/head_specific_candidate_characterization_v1/counterexamples --summary docs/results/head_specific_candidate_characterization_v1.md --output reports/head_specific_candidate_characterization_v1/failure_taxonomy
```

## Result

All primary heads remain falsified. This taxonomy diagnoses why they failed; it does not select new heads.

## Failure Mode Counts

| failure_mode | n_rows |
| --- | --- |
| control_moved | 80 |
| domain_flip | 40 |
| intervention_disagreement | 40 |
| length_flip | 40 |
| position_mismatch | 16 |
| reversed_control_leak | 26 |
| target_swap_leak | 54 |

## Primary Heads

| head_label | final_status | mean_gap | dominant_failure_modes |
| --- | --- | --- | --- |
| L0H8 | falsified_candidate | -0.1949 | control_moved,target_swap_leak,reversed_control_leak,domain_flip,length_flip |
| L7H0 | falsified_candidate | -0.0336 | control_moved,target_swap_leak,reversed_control_leak,domain_flip,length_flip |
| L7H11 | falsified_candidate | -0.1316 | control_moved,target_swap_leak,reversed_control_leak,domain_flip,length_flip |
| L7H7 | falsified_candidate | -0.055 | control_moved,target_swap_leak,reversed_control_leak,domain_flip,length_flip |
| L9H11 | falsified_candidate | -0.0048 | control_moved,target_swap_leak,reversed_control_leak,domain_flip,length_flip |

## By-Head Interpretation

### L0H8

- Final status: `falsified_candidate`
- Dominant failure modes: `control_moved,target_swap_leak,reversed_control_leak,domain_flip,length_flip`
- Interpretation: L0H8 failed because controls moved under the same intervention logic; wrong-target or target-swap controls leaked; reversed-order controls moved; token domain or sequence length changed the effect direction; intervention or position variants disagreed; attention/effect alignment was weak; negative-control support made the slice-level rule too permissive. Its consolidated status was `falsified_candidate`.

### L7H0

- Final status: `falsified_candidate`
- Dominant failure modes: `control_moved,target_swap_leak,reversed_control_leak,domain_flip,length_flip`
- Interpretation: L7H0 failed because controls moved under the same intervention logic; wrong-target or target-swap controls leaked; reversed-order controls moved; token domain or sequence length changed the effect direction; intervention or position variants disagreed; attention/effect alignment was weak; negative-control support made the slice-level rule too permissive. Its consolidated status was `falsified_candidate`.

### L7H11

- Final status: `falsified_candidate`
- Dominant failure modes: `control_moved,target_swap_leak,reversed_control_leak,domain_flip,length_flip`
- Interpretation: L7H11 failed because controls moved under the same intervention logic; wrong-target or target-swap controls leaked; reversed-order controls moved; token domain or sequence length changed the effect direction; intervention or position variants disagreed; attention/effect alignment was weak; OV/QK local diagnostics did not combine with causal specificity; negative-control support made the slice-level rule too permissive. Its consolidated status was `falsified_candidate`.

### L7H7

- Final status: `falsified_candidate`
- Dominant failure modes: `control_moved,target_swap_leak,reversed_control_leak,domain_flip,length_flip`
- Interpretation: L7H7 failed because controls moved under the same intervention logic; wrong-target or target-swap controls leaked; reversed-order controls moved; token domain or sequence length changed the effect direction; intervention or position variants disagreed; attention/effect alignment was weak; OV/QK local diagnostics did not combine with causal specificity; negative-control support made the slice-level rule too permissive. Its consolidated status was `falsified_candidate`.

### L9H11

- Final status: `falsified_candidate`
- Dominant failure modes: `control_moved,target_swap_leak,reversed_control_leak,domain_flip,length_flip`
- Interpretation: L9H11 failed because controls moved under the same intervention logic; wrong-target or target-swap controls leaked; reversed-order controls moved; token domain or sequence length changed the effect direction; intervention or position variants disagreed; attention/effect alignment was weak; OV/QK local diagnostics did not combine with causal specificity; negative-control support made the slice-level rule too permissive. Its consolidated status was `falsified_candidate`.

## Concrete Counterexample Rows

| head_label | failure_mode | family | intervention | position_label | effect_size |
| --- | --- | --- | --- | --- | --- |
| L0H8 | control_moved | char_target_swap_control | head_zero_ablation | final | 3.9638556118503088 |
| L0H8 | control_moved | char_reversed_control | head_zero_ablation | final | 2.329712146819206 |
| L0H8 | control_moved | char_target_swap_control | head_mean_ablation | final | 1.3437189260855154 |
| L0H8 | control_moved | char_reversed_control | head_zero_ablation | final | 0.5033469222103996 |
| L0H8 | control_moved | char_reversed_control | head_zero_ablation | final | 0.2100904040781478 |
| L0H8 | control_moved | char_target_swap_control | head_zero_ablation | final | 0.1971228789402819 |
| L0H8 | control_moved | char_reversed_control | head_mean_ablation | final | 0.1854849193967758 |
| L0H8 | control_moved | char_reversed_control | head_zero_ablation | final | 0.1497878090003429 |
| L0H8 | control_moved | char_target_swap_control | head_zero_ablation | final | 3.9638556118503088 |
| L0H8 | control_moved | char_target_swap_control | head_mean_ablation | final | 1.3437189260855154 |
| L0H8 | control_moved | char_target_swap_control | head_zero_ablation | final | 0.1971228789402819 |
| L0H8 | control_moved | char_target_swap_control | head_zero_ablation | final | 0.140251007468791 |
| L0H8 | control_moved | char_target_swap_control | head_zero_ablation | final | 0.0952704307271398 |
| L0H8 | control_moved | char_target_swap_control | head_mean_ablation | final | 0.0412988504787838 |
| L0H8 | control_moved | char_target_swap_control | head_zero_ablation | final | 0.0317189898612329 |
| L0H8 | control_moved | char_target_swap_control | head_mean_ablation | final | 0.0278847167408407 |
| L0H8 | domain_flip | char_multi_distractor | head_zero_ablation | final | -3.7637761828843774 |
| L0H8 | domain_flip | char_multi_distractor | head_zero_ablation | previous_occurrence | -1.3752958245298528 |
| L0H8 | domain_flip | char_multi_distractor | head_zero_ablation | final | -1.2192680774187132 |
| L0H8 | domain_flip | char_multi_distractor | head_zero_ablation | final | -1.1527696166173866 |

## What This Means

The failure pattern points to calibration work before any future candidate search. Controls moved, target-swap and reversed-order controls leaked, domain and length variants disagreed, and intervention or position slices were not stable enough to support a mechanism claim.

## What This Does Not Show

This does not show an induction head, a circuit, or a broad GPT-2 property. It also does not justify adding new heads. The next step is metric and prompt calibration.

## Exact Next Command

```bash
less docs/results/head_specific_candidate_characterization_failure_taxonomy_v1.md
```
