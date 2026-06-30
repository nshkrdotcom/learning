# Hypothesis: multi_qkv_static_3track_global_30m_seed1

CLAIM:
A globally shared three-track bundled Q/K/V bank with static depth routing changes local training behavior relative to standard attention.

KILL_CONDITION:
The run is unstable, fails verification, emits degenerate QKV diagnostics, or is worse than standard attention without a compensating mechanism signal.

MECHANISM_PROOF:
Diagnostics must show global-bank usage, nonzero selected-track gradients, nonzero selected-track output norms, and expected active_track = layer_idx mod 3 routing.

NEAREST_BORING_EXPLANATION:
Any effect is due to extra Q/K/V capacity or shared-bank implementation artifacts, not a useful schedule.

CONTROL_THAT_RULES_IT_OUT:
standard_refactor_control_30m_seed1.
