# ML Viz Lab

ML Viz Lab is a Phoenix LiveView application for visualizing small machine-learning libraries through synchronized source, graph, and teaching panes.

The first adapter is `MicrogradEx`, loaded from the sibling path dependency `../micrograd_ex`. The app is intentionally not named or structured as a Micrograd-only project: subject-specific code is isolated under `MlVizLab.Subjects.Micrograd`, while the LiveView shell consumes generic lesson, source, concept, and trace JSON.

## Run

```sh
mix setup
mix phx.server
```

Open <http://localhost:4000>.

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

- `MlVizLab.Subjects.Adapter` defines the boundary future subjects must implement.
- `MlVizLab.Trace.*` structs define the subject-neutral trace contract sent to the browser.
- `MlVizLab.Runs` centralizes ephemeral run ids, async trace generation, and error payloads.
- `MlVizLab.Subjects.Micrograd` owns the current lesson catalog and executes real `MicrogradEx` code.
- `MlVizLab.Subjects.Micrograd.TraceBuilder` converts real execution results into immutable timeline events.
- `MlVizLabWeb.VizLive` renders the single-page shell, starts async trace runs, and pushes `trace_ready`.
- `assets/js/viz_lab.js` owns local playback, CodeMirror source highlighting, Three.js graph rendering, node/edge inspection, and teaching-card synchronization.
- `assets/js/viz/` contains testable playback, trace-store, URL, and formatting helpers.

The trace is generated from real BEAM execution, then replayed client-side. Scrubbing, stepping, loop-phase playback, compare mode, and camera follow move through the immutable timeline instead of re-executing code on every cursor change.

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

The tests verify that every lesson generates a typed source-backed trace, key gradients and parameter updates match MicrogradEx, compressed training loss decreases, playback reducers behave correctly, all lessons render in Chromium, and graph canvases are visibly nonblank.

## Future Subjects

A future `Makemore` adapter should be added as a sibling module such as `MlVizLab.Subjects.Makemore`. It should provide its own source catalog, lesson catalog, concept copy, and trace builder while returning the same generic trace shape consumed by the frontend.

## Future LLM Tutor

The teaching pane includes a reserved LLM surface, but no API integration is enabled. The current UI works entirely through lesson selection, cinema controls, source tabs, graph inspection, concept chips, and explanation-depth controls.
