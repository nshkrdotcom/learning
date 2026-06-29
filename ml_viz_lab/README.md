# ML Viz Lab

ML Viz Lab is a Phoenix LiveView application for visualizing small Elixir programs through synchronized source, execution-state, graph, and teaching panes.

The first specimen is `MicrogradEx`, loaded from the sibling path dependency `../micrograd_ex`. The app is intentionally not named or structured as a Micrograd-only project: Micrograd-specific code is isolated under `MlVizLab.Subjects.Micrograd`, while the execution/session layer is generic Elixir AST stepping.

## Run

```sh
mix setup
mix phx.server
```

Open <http://localhost:4000>.

For the live backend stepping demo, open:

<http://localhost:4000/?subject=micrograd&lesson=x_squared&mode=live>

In live mode the first visible action is `Start live`. The bottom cinema Play button is also safe in live mode: before a session exists it starts live execution instead of sending `continue_live`.

## Installed Tooling And Dependencies

This project was created with Phoenix 1.8.8. During implementation, these machine-level tools were installed or refreshed:

- Hex was refreshed with `mix local.hex --force`.
- Phoenix generator archive was installed with `mix archive.install hex phx_new --force`, producing `phx_new 1.8.8` under the active Elixir installation.
- Playwright Chromium browser binaries were installed with `npx --prefix assets playwright install chromium`; Playwright stores those under `~/.cache/ms-playwright/`.

These project-local npm packages are recorded in `assets/package.json` and `assets/package-lock.json`:

- Runtime: `three`, `dagre`, `codemirror`, `codemirror-lang-elixir`, `@codemirror/language`, `@codemirror/state`, `@codemirror/view`.
- Dev/test: `playwright`, `vitest`, `pngjs`.

Use `mix setup` for Elixir dependencies and npm install. Use `npx --prefix assets playwright install chromium` if browser smoke tests fail because the Playwright browser cache is missing on a different machine.

## Architecture

- `MlVizLab.Execution.*` is the primary runtime architecture. It parses configured lesson source with `Code.string_to_quoted/2`, instruments top-level expressions, starts an isolated runtime process, pauses before source spans, captures bindings after each expression, and advances only when LiveView sends a backend command.
- `MlVizLab.Execution.Session` owns one live execution. It tracks `session_id`, `subject_id`, `lesson_id`, status, runtime pid, current step/span, bindings, events, timestamps, and errors.
- `MlVizLab.Execution.RuntimeHooks.pause/3` is the real lockstep boundary. The runtime process sends a paused event to the controller and blocks in `receive` until it gets `:continue` or `:stop`.
- `MlVizLab.Execution.SafeInspect` and `BindingSnapshot` serialize live bindings safely; raw BEAM values are not sent to the browser.
- `MlVizLab.Subjects.Micrograd.LiveLesson` provides configured raw MicrogradEx lesson source. The live `x_squared` path calls `MicrogradEx.Value` directly; it does not use Micrograd wrapper modules.
- `MlVizLab.Subjects.Micrograd.DomainSnapshot` observes live bindings and derives Micrograd graph/gradient state for the visual pane.
- `MlVizLabWeb.VizLive` supports both `mode=live` and replay mode. In live mode, controls send `start_live`, `step_live`, `continue_live`, `stop_live`, and `reset_live`; stale session events are ignored.
- `assets/js/viz_lab.js` has separate paths for `applyExecutionEvent(event)` in live mode and `loadTrace(trace)` in replay mode. CodeMirror and graph-owned DOM are isolated from LiveView patches.
- `MlVizLab.Runs`, `MlVizLab.Trace.*`, `MlVizLab.Instrumentation.*`, and the Micrograd instrumented runner remain as recorded replay/compatibility infrastructure.

Live AST execution is not raw BEAM instruction stepping. It is source-level semantic stepping over configured scripts. BEAM tracing may later supplement process/message visibility, but source spans and local bindings come from AST instrumentation.

Recorded replay remains useful after or beside live execution: immutable traces support smooth scrub/rewind, loop-phase playback, compare mode, and camera follow. The UI labels live execution and replay separately.

## Lessons

The initial Micrograd subject ships twelve base lessons:

- `x_squared`
- `sum_rule`
- `chain_rule_composed`
- `karpathy_sanity`
- `repeated_parent`
- `relu_gate`
- `single_neuron`
- `one_layer_forward`
- `tiny_mlp`
- `mlp_loss_backward`
- `one_training_step`
- `linear_training`

The `linear_training` lesson is intentionally compressed: each event summarizes one real training step so loss and parameter movement are visible without thousands of scalar graph events.

## Correctness

Run:

```sh
mix test
mix assets.build
npm test --prefix assets
```

With a Phoenix server running on `localhost:4000`, run browser smoke coverage:

```sh
npm run test:e2e --prefix assets
```

Use `window.VIZ_LIVE_DEBUG = true` in the browser console before interacting with the live page to enable frontend debug logs prefixed with `[ml-viz-live]`.

The tests verify that every lesson generates a typed source-backed trace, key gradients and parameter updates match MicrogradEx, compressed training loss decreases, playback reducers behave correctly, all lessons render in Chromium, and graph canvases are visibly nonblank.

The live execution tests additionally prove that `x_squared` is paused in the backend before the first line executes, `Next` unblocks the runtime for one expression, bindings appear progressively (`x`, then `y`, then `gradients`), and the browser graph is not preloaded with future state.

## Future Subjects

A future `Makemore` adapter should be added as a sibling module such as `MlVizLab.Subjects.Makemore`. For live execution, it should provide configured lesson source plus an optional domain snapshot adapter. For replay, it may also provide sources, concepts, and trace generation.

To add a subject:

1. Implement `MlVizLab.Subjects.Adapter`.
2. Return generic metadata from `id/0`, `title/0`, `description/0`, and `capabilities/0`.
3. Return lessons, concepts, and sources without Micrograd-specific assumptions.
4. Implement `run/2` so it accepts `:run_id` and returns `{:ok, %MlVizLab.Trace.Run{}}` or `{:error, reason}`.
5. Optionally expose `live_source/1` and `live_domain_adapter/0` for live AST mode.
6. Register it in `config :ml_viz_lab, :subjects`.

`MlVizLab.Runs` and `VizLive` should not need changes for a new subject.

## Adding Micrograd Lessons

Add lesson metadata in `MlVizLab.Subjects.Micrograd.lessons/0`. For live execution, add raw lesson source in `MlVizLab.Subjects.Micrograd.LiveLesson`; the source should call `MicrogradEx` directly. If replay coverage is also needed, add or update the compatibility trace runner.

After adding a lesson, run:

```sh
mix test test/ml_viz_lab/subjects/micrograd_test.exs
mix assets.build
npm test --prefix assets
```

Live authenticity is checked by backend pause/continue tests and progressive binding/domain snapshots. Replay authenticity is checked by comparing replayed gradient propagation with `MicrogradEx.Value.backward/1`, verifying source spans against bundled source text, and asserting known numeric gradients/updates.

## Future LLM Tutor

The teaching pane includes a reserved LLM surface, but no API integration is enabled. The current UI works entirely through lesson selection, cinema controls, source tabs, graph inspection, concept chips, and explanation-depth controls.
