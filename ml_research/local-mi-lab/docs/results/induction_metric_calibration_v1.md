# Induction Metric Calibration v1

## Source

- Spec: `docs/experiments/induction_metric_calibration_v1.md`
- Command: `uv run python scripts/run_metric_calibration.py --output reports/induction_metric_calibration_v1`
- Model: `gpt2-small`

## Executive Summary

The metric did not pass the pre-registered separation thresholds.

## Primary Metric

`true_vs_control_logit_diff`.

## Thresholds

| threshold | value |
| --- | --- |
| min_positive_minus_max_control_gap | 0.25 |
| min_positive_fraction_diff_positive | 0.8 |
| max_control_fraction_diff_positive | 0.2 |

## Overall Separation

- Positive mean: `4.502581834793091`
- Max control mean: `3.2170213063557944`
- Positive-minus-max-control gap: `1.2855605284372964`
- Weakest positive family mean: `1.7427806854248047`
- Hardest control family: `{'family': 'calib_frequency_trap_control', 'mean_true_vs_control_logit_diff': 3.2170213063557944, 'fraction_diff_positive': 1.0}`

## Family Summary

| family | should_show_induction_behavior | mean_true_vs_control_logit_diff | fraction_diff_positive | median_target_rank |
| --- | --- | --- | --- | --- |
| calib_clean_repeat_format_variant | True | 6.275806427001953 | 1.0 | 1.0 |
| calib_clean_repeat_number | True | 4.7462952931722 | 1.0 | 1.0 |
| calib_clean_repeat_symbolic | True | 5.245444933573405 | 1.0 | 1.0 |
| calib_clean_repeat_word | True | 1.7427806854248047 | 1.0 | 1.0 |
| calib_frequency_trap_control | False | 3.2170213063557944 | 1.0 | 1.0 |
| calib_no_repeat_control | False | 0.0834705564710829 | 0.6666666666666666 | 13.0 |
| calib_reversed_order_control | False | -1.0967043770684137 | 0.2222222222222222 | 11.0 |
| calib_same_token_frequency_control | False | -0.22988933987087673 | 0.4444444444444444 | 5.0 |
| calib_target_swap_control | False | -3.5317175123426647 | 0.0 | 5.0 |
| calib_wrong_target_same_prompt | False | -3.91150697072347 | 0.0 | 5.0 |

## Domain and Length Checks

- Positive domain means: `{'number': 6.0308332443237305, 'symbolic': 6.323929309844971, 'word': 2.9262075424194336}`
- Positive length means: `{'long': 7.492025693257649, 'medium': 4.369056224822998, 'short': 3.419888178507487}`

## Decision

Final status: `metric_needs_revision`.
Search allowed: `False`.

## Interpretation

Do not search for heads. The metric is still false-positive-prone or insufficiently separated from controls.

## What This Does Not Show

This does not show an induction head, a circuit, or a broad GPT-2 property. It only tests whether one local behavior metric is calibrated enough for future specifications.

## Exact Next Command

```bash
less docs/results/induction_metric_calibration_v1.md
```
