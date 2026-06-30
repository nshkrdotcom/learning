# Hypothesis: multi_qkv_position_rotation_3track_global_30m_seed1

CLAIM:
Position-dependent deterministic routing changes local training/eval behavior beyond static depth routing.

KILL_CONDITION:
The run is unstable, fails verification, emits degenerate or missing position-routing diagnostics, or is no better than static routing with materially worse throughput/VRAM.

MECHANISM_PROOF:
Diagnostics must show expected active_track = (layer_idx + pos) mod 3 during train and eval, all tracks active over typical contexts, and nonzero selected-track gradients.

NEAREST_BORING_EXPLANATION:
Any effect is due to extra three-track capacity or global-bank sharing, not position routing.

CONTROL_THAT_RULES_IT_OUT:
multi_qkv_static_3track_global_30m_seed1 and standard_refactor_control_30m_seed1.
