# 0901 Rebuild 01: Experiment Math and Variant Contract

## Shared Structure

All first-build variants use:

```text
track_count = 3
Q = Q_bank[active_track]
K = K_bank[active_track]
V = V_bank[active_track]
```

Q/K/V are bundled. The same `active_track` selects all three projections.

Because the bank is globally shared across layers, the first-build Multi-QKV variants do not have the same parameter count as standard attention. This is intentional and must be reported in comparisons. B and C should be compared primarily against A because A has the same global-bank parameterization without train-step or position rotation.

## A: Static Depth-Only Cycle

```text
active_track(layer_idx, step, pos) = layer_idx mod 3
```

This is the primary extra-capacity/global-sharing control.

## B: Train-Time Step Rotation, Frozen Deployment

Training:

```text
active_track(layer_idx, step, pos) = (layer_idx + step) mod 3
```

Eval/generation:

```text
active_track(layer_idx, step, pos) = layer_idx mod 3
```

Missing `step` during training is an error. Eval/generation must not use training-step rotation.

## C: Position Rotation

Training and eval/generation:

```text
active_track(layer_idx, step, pos) = (layer_idx + pos) mod 3
```

This is a per-position hard switch. Token position is the absolute position inside the current model context. This repo does not use KV-cache generation, so generation recomputes the visible context and positions are derived from that context.

## Initial Non-Goals

```text
- No learned softmix.
- No warmup/annealed routing.
- No LoRA deltas.
- No decoupled Q/K/V clocks.
- No stochastic schedule.
- No typed content/operator/binding streams.
```

## Future Work Only

Future variants may test learned softmix, warmup schedules, LoRA deltas, decoupled Q/K/V clocks, stochastic schedules, or typed streams, but they are not part of E002 first-build A/B/C.
