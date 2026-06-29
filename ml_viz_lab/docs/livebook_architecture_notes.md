# Livebook Architecture Notes

Livebook is useful as an architectural reference, not as a direct dependency for
this phase.

Observed reference points:

- Livebook is a Phoenix web application for interactive notebooks. Its README
  describes on-demand Elixir code evaluation, CodeMirror editing, Kino-powered
  rich outputs, reproducible execution order, stale-state tracking, and custom
  runtimes.
- Livebook runtimes are sets of processes that evaluate notebook code. The
  documented runtime choices include standalone runtime nodes, attached nodes,
  Fly.io runtimes, and Kubernetes pod runtimes.
- The attached runtime proves the model we care about: a web UI can control code
  evaluation in an existing BEAM node with access to the project modules.
- Livebook's package documentation states that the `livebook` package is meant
  as a CLI tool and is not officially supported as a Mix/Hex dependency.
- Livebook sessions use a central session process and pure session-data
  transitions that return actions. Side effects such as actual evaluation are
  started by the session process after state transitions.
- Kino is valuable future inspiration for rich outputs and interactive widgets,
  but it is not needed for the first backend pause/continue runtime.

Decision:

- Do not embed Livebook wholesale.
- Build a small project-local runtime focused on AST source stepping.
- Reuse the architectural shape: session process, runtime process, web events,
  explicit runtime state, and output/event records.
- Keep arbitrary user code execution out of scope. Live execution uses
  configured lesson sources only.

Sources inspected:

- https://hexdocs.pm/livebook/
- https://hexdocs.pm/livebook/runtime.html
- https://github.com/livebook-dev/livebook
- https://github.com/livebook-dev/livebook/blob/main/lib/livebook/session/data.ex
