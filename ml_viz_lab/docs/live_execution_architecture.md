# Live Execution Architecture

ML Viz Lab is a browser-controlled BEAM execution microscope. The primary path is live backend execution, not prerecorded replay.

## Runtime Path

```text
Browser
  <-> Phoenix LiveView / VizLab hook
  <-> MlVizLab.Execution.Session
  <-> MlVizLab.Execution.Runtime
  <-> AST-instrumented lesson source
  <-> real target modules such as MicrogradEx.Value
```

`MlVizLabWeb.VizLive` owns browser-facing state and routes commands to the current session. `MlVizLab.Execution.Session` owns one live run and normalizes runtime events for LiveView. `MlVizLab.Execution.Runtime` evaluates configured source in a separate process. `MlVizLab.Execution.RuntimeHooks.pause/3` is the lockstep boundary: it sends a paused event and blocks until the session sends continue or stop.

## Source And Spans

Configured lesson source is parsed and instrumented by `MlVizLab.Execution.AstInstrumenter`. For the supported straight-line lesson subset, each top-level expression is wrapped as:

```elixir
MlVizLab.Execution.RuntimeHooks.pause(runtime, span, binding())
result = original_expression
MlVizLab.Execution.RuntimeHooks.capture(runtime, span, binding(), result)
result
```

This keeps execution real, preserves assignment semantics, and captures bindings only after the backend has actually evaluated an expression.

## Live State Contract

Live mode has its own state independent of replay traces:

- `live_session_id`
- `live_session_pid`
- `live_runtime_pid`
- `live_status`
- `live_current_span`
- `live_step_index`
- `live_source`
- `live_bindings`
- `live_domain_snapshot`
- `live_events`
- `live_error`
- `live_generation`

The frontend treats `trace === null` as valid in live mode. Command errors update live error state and never clear source. Events with stale session ids are ignored.

## MicrogradEx Specimen

MicrogradEx is the first constrained specimen, not the architecture boundary. Live source lives in `MlVizLab.Subjects.Micrograd.LiveLesson` and calls `MicrogradEx.Value` directly. `MlVizLab.Subjects.Micrograd.DomainSnapshot` derives graph nodes, edges, scalar data, and gradients from real live bindings.

For `x_squared`, the live progression is:

- before start: source visible, no backend session
- first pause: graph empty, `x` not yet bound
- after first step: `x` exists
- after second step: `y` exists with data `9`
- after backward: gradients include `x = 6` and `y = 1`

## Replay

Replay remains secondary infrastructure for immutable trace playback, scrub/rewind, compare mode, and recorded demonstrations. Replay code should consume `MlVizLab.Trace.*`; live execution should consume `MlVizLab.Execution.*` events and optional subject domain snapshots.

## Current Limits

- AST instrumentation intentionally supports configured straight-line lesson scripts, not arbitrary untrusted browser code.
- Live mode is backend-controlled and cannot rewind without reset/re-run.
- The teaching panel is scripted status/explanation UI. There is no LLM/API integration.
