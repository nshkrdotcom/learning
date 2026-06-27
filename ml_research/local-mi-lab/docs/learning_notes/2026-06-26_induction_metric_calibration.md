# Induction Metric Calibration: Practice Note

## Question

Can `true_vs_control_logit_diff` separate repeated-token positives from controls before another head search?

## Result

Final status: `metric_needs_revision`.

## What Worked

Positives exceeded the hardest control by `1.2855605284372964` on average.

## What Failed

The hardest control `calib_frequency_trap_control` had mean diff `3.2170213063557944`, so controls remain central to interpretation.

## Hardest Control

`{'family': 'calib_frequency_trap_control', 'mean_true_vs_control_logit_diff': 3.2170213063557944, 'fraction_diff_positive': 1.0}`

## What I Learned

Metric calibration is a prerequisite for candidate search. A good-looking intervention row is not useful if controls score under the same rule.

## What I Will Not Claim

I will not claim an induction head, a circuit, or broad GPT-2 behavior from this calibration pass.
