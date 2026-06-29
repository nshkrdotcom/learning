# Live Execution Runtime

The previous architecture generated an immutable Micrograd trace and replayed it
in the browser. That remains useful as recorded replay mode, but it is not true
backend lockstep.

The corrected architecture adds live AST execution:

```text
Browser LiveView / JS hook
  -> Execution session controller
  -> Runtime process
  -> AST-instrumented configured lesson source
  -> Real target modules
  -> Optional domain snapshot adapter
```

Live execution means:

- The lesson source is parsed with `Code.string_to_quoted/2`.
- The top-level expressions are instrumented with runtime hook calls.
- The runtime process pauses before each top-level expression.
- A frontend command sends `step_live` or `continue_live`.
- The controller forwards a continue command to the paused runtime process.
- The runtime executes exactly the next expression, captures bindings, then
  pauses again at the next source span.

The runtime is deliberately source-level and semantic. It is not raw BEAM
instruction stepping. BEAM tracing can later add process/message authenticity,
but AST instrumentation is what provides source spans and local bindings.

For Micrograd, the live lesson source calls raw `MicrogradEx.Value` functions.
The Micrograd domain adapter observes live bindings and derives graph/gradient
state from `MicrogradEx.Value` and `MicrogradEx.Gradients` values. Micrograd is
therefore a specimen under the runtime, not the runtime itself.

Recorded replay mode remains separate:

- Replay traces are immutable and scrub-friendly.
- Live mode is command-driven and cannot rewind without resetting/re-running.
- The UI labels live execution and replay distinctly.

Future concurrent BEAM support requires more structure:

- process tree and runtime pid tracking
- causal ids for commands/events
- process timelines
- mailbox/message events
- scheduler/step policies
- BEAM trace integration for send/receive/spawn/call/cast reality

This phase implements single-process AST stepping first.
