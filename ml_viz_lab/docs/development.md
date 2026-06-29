# Development

## Local Setup

```sh
cd ~/p/g/n/learning/ml_viz_lab
mix setup
mix phx.server
```

The app depends on the sibling path dependency `../micrograd_ex`. Do not modify that repo for UI behavior; derive visualization state from its public structs where possible.

## Quality Checks

Run the full local suite before committing:

```sh
mix format
mix test
mix assets.build
npm test --prefix assets
npm run test:e2e --prefix assets
```

`npm run test:e2e --prefix assets` requires a Phoenix server at `http://localhost:4000`.

If Playwright cannot find Chromium on a fresh machine:

```sh
npx --prefix assets playwright install chromium
```

## Live Execution Responsibilities

- `MlVizLab.Execution.AstInstrumenter`: parse configured source and wrap supported top-level expressions with pause/capture hooks.
- `MlVizLab.Execution.Runtime`: evaluate instrumented source in a managed process.
- `MlVizLab.Execution.RuntimeHooks`: block and resume runtime execution at source boundaries.
- `MlVizLab.Execution.Session`: own one live run, track status, ignore stale events, and normalize events for LiveView.
- `MlVizLabWeb.VizLive`: keep live assigns separate from replay assigns, push normalized events, and reject invalid commands safely.

## Frontend Responsibilities

- `assets/js/viz/live_state.js`: define the live state reducer and `canSendLiveAction`.
- `assets/js/viz_lab.js`: keep live rendering separate from replay rendering, own CodeMirror and graph DOM, preserve last-known-good source, and push only valid live commands.
- `assets/js/viz/playback.js`: replay-only timeline state.
- `assets/js/viz/trace_store.js`: replay source/trace indexing.

## Tests To Add With New Work

- Backend runtime/session tests for every new command or event type.
- Subject adapter tests for new live domain snapshots.
- JS reducer tests for new live state transitions.
- E2E coverage when a browser-visible workflow changes.

## Known Limits

- Live execution supports configured source only; browser-provided arbitrary code is not evaluated.
- Current AST instrumentation is source-level top-level expression stepping.
- Live reset starts a fresh run; live rewind is not supported.
- The LLM panel is reserved UI only.
