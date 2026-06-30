# Hypothesis: multi_qkv_train_rotation_3track_global_30m_seed1

CLAIM:
Training-time deterministic step rotation pressures the shared Q/K/V bank toward phase-portable behavior beyond static depth routing.

KILL_CONDITION:
The run is unstable, fails verification, emits degenerate QKV diagnostics, remains no better than static routing, or eval/generation accidentally use training-step routing.

MECHANISM_PROOF:
Diagnostics must show expected active_track = (layer_idx + step) mod 3 during training, nonzero selected-track gradients, and eval-freeze behavior in eval/generation.

NEAREST_BORING_EXPLANATION:
Any effect is due to extra three-track capacity or global-bank sharing, not step rotation.

CONTROL_THAT_RULES_IT_OUT:
multi_qkv_static_3track_global_30m_seed1 and standard_refactor_control_30m_seed1.
