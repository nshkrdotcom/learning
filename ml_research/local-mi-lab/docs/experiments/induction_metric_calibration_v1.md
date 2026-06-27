# Induction Metric Calibration v1

## Status

Pre-registered calibration experiment.

This experiment follows `Head-Specific Candidate Characterization v1`, where all 16 fixed heads were classified as `falsified_candidate`. The purpose is not to find a new head. The purpose is to test whether the current prompt and metric pipeline is calibrated enough to justify a future candidate search.

## Question

Does the primary metric, `true_vs_control_logit_diff`, separate clean repeated-token induction-style prompts from controls across token domains, sequence lengths, and harmless formatting changes?

## Metric Under Test

Primary metric:

```text
true_vs_control_logit_diff = logit(true_expected_next_token) - logit(wrong_or_control_token)
```

Secondary metrics:

- `target_logit`
- `control_logit`
- `probability_gap`
- `target_rank`

The secondary metrics are diagnostic only. They cannot rescue a failure of the primary metric.

## Model

Use only:

```text
gpt2-small
```

This is a behavior/metric calibration pass, not a head-intervention pass.

## Prompt Families

The calibration prompt set is deliberately small, hand-inspectable, and balanced across domains.

Positive families:

- `calib_clean_repeat_symbolic`
- `calib_clean_repeat_word`
- `calib_clean_repeat_number`
- `calib_clean_repeat_format_variant`

Control families:

- `calib_wrong_target_same_prompt`
- `calib_target_swap_control`
- `calib_same_token_frequency_control`
- `calib_reversed_order_control`
- `calib_no_repeat_control`
- `calib_frequency_trap_control`

Each family is evaluated across short, medium, and long sequence buckets where possible.

## Expected Behavior

Expected pass behavior:

- clean repeat positives should have higher `true_vs_control_logit_diff` than controls;
- wrong-target same-prompt controls should be clearly lower than positives;
- target-swap controls should not look as good as positives;
- same-token-frequency controls should not be rewarded merely for token counts;
- reversed-order controls should not look like positives;
- symbolic, word, and number domains should not flip the sign of the positive-vs-control gap;
- harmless spacing or separator variants should not reverse the metric direction.

Expected failure behavior:

- controls have mean `true_vs_control_logit_diff` close to or above positives;
- target-swap or wrong-target controls produce positive-looking rows;
- frequency controls are rewarded as strongly as positives;
- one token domain or length bucket drives the entire apparent separation;
- negative-control rows produce “good-looking” metric values under the same threshold.

## Thresholds

This calibration uses conservative descriptive thresholds:

- `mean_positive_minus_max_control_gap > 0.25` logit units is required for a provisional pass;
- every positive family must have positive mean `true_vs_control_logit_diff`;
- no control family may exceed the weakest positive family mean;
- at least 80% of positive examples should have `true_vs_control_logit_diff > 0`;
- no more than 20% of control examples should have `true_vs_control_logit_diff > 0`.

If these thresholds fail, the metric is not calibrated enough for a new head search.

## Tokenization Checks

Every row must explicitly validate:

- `expected_token` is exactly one token;
- `control_token` is exactly one token;
- `expected_token != control_token`;
- prompt text tokenizes without an empty sequence.

Rows that fail tokenization are excluded from metric scoring and make the calibration status `blocked_tokenization` unless they are explicitly marked unsupported.

## Controls

Controls are part of the pass/fail decision, not a secondary appendix. A metric that creates attractive rows for controls is considered false-positive-prone even if some positives also score well.

## Decision Rules

Allowed final statuses:

- `metric_calibrated_for_next_spec`
- `metric_needs_revision`
- `prompt_bank_needs_revision`
- `blocked_tokenization`

Decision logic:

- If tokenization validation fails for required rows, status is `blocked_tokenization`.
- If positives do not separate from controls by threshold, status is `metric_needs_revision`.
- If only one token domain or sequence length works, status is `prompt_bank_needs_revision`.
- Only if all thresholds pass may the status be `metric_calibrated_for_next_spec`.

## What Would Make the Metric Unusable for Candidate Search

The metric is not usable for new candidate search if:

- wrong-target or target-swap controls score similarly to positives;
- same-token-frequency controls score similarly to positives;
- positive-vs-control separation depends on a single token domain;
- the sign flips across short, medium, and long sequences;
- control rows produce attractive scores at a rate comparable to positives.

## Outputs

Expected artifacts:

- `reports/induction_metric_calibration_v1/metric_calibration_by_example.csv`
- `reports/induction_metric_calibration_v1/metric_calibration_by_family.csv`
- `reports/induction_metric_calibration_v1/metric_calibration_summary.json`
- `reports/induction_metric_calibration_v1/induction_metric_calibration_v1.md`
- `docs/results/induction_metric_calibration_v1.md`
- `docs/learning_notes/2026-06-26_induction_metric_calibration.md`

## Refused Claims

Calibration success would not show an induction head, a circuit, or a broad GPT-2 property. It would only show that this local metric/prompt setup is less obviously false-positive-prone than the previous setup.

Calibration failure is an acceptable result. If this metric cannot separate positives from controls, the next step is metric or prompt redesign, not head search.

## Reproduction Command

```bash
uv run python scripts/run_metric_calibration.py \
  --output reports/induction_metric_calibration_v1
```
