# AST Instrumentation

The first instrumenter intentionally supports top-level expression stepping.

Supported inputs:

- aliases/imports in configured lesson source
- assignments
- function calls
- pipelines
- arithmetic expressions
- final expression return values

Instrumentation shape:

```elixir
MlVizLab.Execution.RuntimeHooks.pause(runtime, span, binding())
__viz_value__ = original_expression
MlVizLab.Execution.RuntimeHooks.capture(runtime, span, binding(), __viz_value__)
__viz_value__
```

The injected value variable is filtered out of binding snapshots. Each span
contains a stable expression id, file id, line range, column where available,
kind, and title.

Safety model:

- Only configured lesson source is evaluated.
- No browser-provided arbitrary code is accepted.
- Bindings are serialized through `MlVizLab.Execution.SafeInspect`; raw values
  stay inside the BEAM and are only used for domain adapters.
