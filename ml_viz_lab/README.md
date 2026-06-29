# ML Viz Lab

ML Viz Lab is a Phoenix LiveView application for visualizing small machine-learning libraries through synchronized source, graph, and teaching panes.

The first adapter is `MicrogradEx`, loaded from the sibling path dependency `../micrograd_ex`. The app is intentionally not named or structured as a Micrograd-only project: subject-specific code is isolated under `MlVizLab.Subjects.Micrograd`, while the LiveView shell consumes generic lesson, source, concept, and trace JSON.

## Run

```sh
mix setup
mix phx.server
```

Open <http://localhost:4000>.

## Architecture

- `MlVizLab.Subjects.Adapter` defines the boundary future subjects must implement.
- `MlVizLab.Subjects.Micrograd` owns the current lesson catalog and executes real `MicrogradEx` code.
- `MlVizLab.Subjects.Micrograd.TraceBuilder` converts real execution results into immutable timeline events.
- `MlVizLabWeb.VizLive` renders the single-page shell and embeds the current trace payload.
- `assets/js/viz_lab.js` owns local playback, CodeMirror source highlighting, Three.js graph rendering, node/edge inspection, and teaching-card synchronization.

The trace is generated from real BEAM execution, then replayed client-side. Scrubbing, stepping, and playback move through the immutable timeline instead of re-executing code on every cursor change.

## Lessons

The initial Micrograd subject ships eight base lessons:

- `x_squared`
- `karpathy_sanity`
- `repeated_parent`
- `relu_gate`
- `single_neuron`
- `tiny_mlp`
- `one_training_step`
- `linear_training`

The `linear_training` lesson is intentionally compressed: each event summarizes one real training step so loss and parameter movement are visible without thousands of scalar graph events.

## Correctness

Run:

```sh
mix test
mix assets.build
```

The tests verify that every lesson generates a source-backed trace, key gradients and parameter updates match MicrogradEx, and compressed training loss decreases.

## Future Subjects

A future `Makemore` adapter should be added as a sibling module such as `MlVizLab.Subjects.Makemore`. It should provide its own source catalog, lesson catalog, concept copy, and trace builder while returning the same generic trace shape consumed by the frontend.

## Future LLM Tutor

The teaching pane includes a reserved LLM surface, but no API integration is enabled. The current UI works entirely through lesson selection, cinema controls, source tabs, graph inspection, concept chips, and explanation-depth controls.
