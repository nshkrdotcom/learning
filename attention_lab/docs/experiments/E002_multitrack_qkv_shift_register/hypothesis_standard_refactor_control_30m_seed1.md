# Hypothesis: standard_refactor_control_30m_seed1

CLAIM:
This run establishes the standard-attention refactor/control comparison point under the E002 fixed training contract.

KILL_CONDITION:
The run fails manifest verification, train/eval/reload verification, or produces unstable losses.

MECHANISM_PROOF:
Standard attention has no non-standard mechanism activity requirement; validity is established by manifest checks, full training completion, eval_loss reload, summarize_run, and verify_run.

NEAREST_BORING_EXPLANATION:
Any difference from the historical baseline is due to source-state drift, run variance, or operator environment changes.

CONTROL_THAT_RULES_IT_OUT:
Historical baseline_15m/baseline_30m run and this standard refactor control under the E002 fixed contract.
