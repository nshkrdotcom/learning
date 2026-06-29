# Live Execution Completion Work Log - 2026-06-28

## Detailed Checklist

### Phase 0 - Inspect Current Code and Establish Baseline

- [x] Inspect `lib/ml_viz_lab_web/live/viz_live.ex`.
- [x] Inspect `lib/ml_viz_lab/execution/*.ex`.
- [x] Inspect `lib/ml_viz_lab/subjects/micrograd/*.ex`.
- [x] Inspect app supervision, router, config, and tests.
- [x] Inspect `assets/js/viz_lab.js`.
- [x] Inspect `assets/js/viz/live_state.js`.
- [x] Inspect playback, trace store, code, graph, hook, asset tests, and e2e tests.
- [x] Inspect local dependency `../micrograd_ex` value, gradients, NN modules, and tests.
- [x] Run baseline `mix format --check-formatted || true`.
- [x] Run baseline `mix test || true`.
- [x] Run baseline `mix assets.build || true`.
- [x] Run baseline `npm test --prefix assets || true`.
- [x] Run baseline `npm run test:e2e --prefix assets || true`.

### Phase 1 - Reproduce Browser Bug and Instrument Path

- [x] Run `mix phx.server`.
- [x] Open `/?subject=micrograd&lesson=x_squared&mode=live`.
- [x] Confirm source is visible immediately.
- [x] Click bottom Play before Start Live.
- [x] Observe source panel, browser console, and Phoenix logs.
- [x] Add backend live-flow logs where missing.
- [x] Add frontend debug logging behind `window.VIZ_LIVE_DEBUG === true`.

### Phase 2 - Define Live Execution State Contract

- [x] Normalize backend live assigns.
- [x] Normalize frontend live state.
- [x] Ensure `trace === null` is valid in live mode.
- [x] Ensure command errors update `live.error` without clearing source.
- [x] Ensure stale live events do not mutate state.
- [x] Add state contract tests.

### Phase 3 - Fix Live Controls and Command Gating

- [x] Implement authoritative `canSendLiveAction(live, action)`.
- [x] Apply gating to explicit live buttons.
- [x] Apply gating to bottom cinema controls.
- [x] Apply gating to keyboard shortcuts.
- [x] Server-side structured command errors.
- [x] Regression tests for Play/Step before a session and source preservation.

### Phase 4 - Harden Runtime Process and Pause/Continue Semantics

- [x] Verify live run session ids are unique.
- [x] Verify supervised sessions start runtime processes.
- [x] Verify runtime blocks at pause points.
- [x] Verify step advances one boundary.
- [x] Verify stop terminates runtime cleanly.
- [x] Verify exceptions become structured events.
- [x] Verify stale commands/events are ignored.

### Phase 5 - AST Instrumentation Completion

- [x] Verify supported subset and source span behavior.
- [x] Verify assignments capture new bindings.
- [x] Verify comments and blank lines do not create bogus spans.
- [x] Verify invalid source errors are structured.

### Phase 6 - MicrogradEx Live Domain Adapter

- [x] Verify domain snapshots derive graph data from real `MicrogradEx.Value` structs.
- [x] Verify x appears before y.
- [x] Verify y data is 9.
- [x] Verify backward exposes x gradient 6 and y gradient 1.

### Phase 7 - CodeMirror Code View Completion

- [x] Ensure CodeMirror initializes once per hook mount.
- [x] Keep LiveView from morphing editor internals.
- [x] Preserve last-known-good source across bad events.
- [x] Update active span without recreating editor unnecessarily.
- [x] Add code view tests.

### Phase 8 - Live Graph View Completion

- [x] Render empty pre-run graph state.
- [x] Render live graph from explicit domain snapshot.
- [x] Show nodes, edges, values, gradients, and active state progressively.
- [x] Add graph tests.

### Phase 9 - Teaching / Status Panel

- [x] Implement synchronized scripted teaching/status copy.
- [x] Cover before run, before/after x, before/after y, before/after backward, and completed.
- [x] Show command errors without clearing source.

### Phase 10 - LiveView Template and DOM Ownership

- [x] Add stable `data-testid` selectors.
- [x] Add real disabled states and accessible labels.
- [x] Isolate CodeMirror and graph-owned DOM where needed.
- [x] Keep code, graph, and teaching panels mounted.

### Phase 11 - Browser / E2E Verification

- [x] Add live `x_squared` happy path test.
- [x] Add invalid live command does not blank code test.
- [x] Add reset/restart stale event coverage where practical.

### Phase 12 - Local Developer Workflow Docs

- [x] Update `README.md`.
- [x] Create/update `docs/live_execution_architecture.md`.
- [x] Create/update `docs/live_mode_repro.md`.
- [x] Create/update `docs/development.md`.

### Phase 13 - Testing and QC

- [x] Run `mix format`.
- [x] Run `mix test`.
- [x] Run `mix assets.build`.
- [x] Run `npm test --prefix assets`.
- [x] Run `npm run test:e2e --prefix assets`.
- [x] Manual browser verification for live and replay URLs.

### Phase 14 - Commit and Push

- [x] Review `git status` and `git diff`.
- [x] Commit `Complete live execution microscope flow`.
- [x] Push.

## Baseline Commands

- `mix format --check-formatted || true`
  - Result: passed.
- `mix test || true`
  - Result: passed, 46 tests.
- `mix assets.build || true`
  - Result: passed. Tailwind and esbuild completed.
- `npm test --prefix assets || true`
  - Result: passed, 3 files / 13 tests.
- `npm run test:e2e --prefix assets || true`
  - Result: failed because no server was listening on `http://localhost:4000`.
  - Error: `page.goto: net::ERR_CONNECTION_REFUSED`.

## Current Architecture Notes

- Live mode is already routed through `MlVizLabWeb.VizLive`, `MlVizLab.Execution.Session`, `MlVizLab.Execution.Runtime`, `MlVizLab.Execution.RuntimeHooks`, and `MlVizLab.Execution.AstInstrumenter`.
- Runtime execution is real: `RuntimeHooks.pause/3` sends a paused event and blocks in `receive` until continue or stop.
- Micrograd live lessons use raw `MicrogradEx.Value` calls in `MlVizLab.Subjects.Micrograd.LiveLesson`.
- Micrograd graph state is already derived from real live bindings by `MlVizLab.Subjects.Micrograd.DomainSnapshot`.
- Replay and live share one large frontend hook. Several replay assumptions still leak into live controls and rendering.
- CodeMirror is currently recreated in several paths and no explicit last-known-good source guard exists.
- Bottom cinema controls map Play to `continue_live` in live mode even when no session exists.

## Known Broken Paths

- In live mode, bottom Play before Start Live can send `continue_live` with no session.
- Server command errors use a generic event shape and do not preserve a clear live status contract.
- Frontend command paths do not have one authoritative gate.
- Live source preservation is implicit; malformed events can still make render paths fragile.
- LiveView template lacks several stable selectors requested for browser tests.
- E2E coverage currently only smoke-tests replay and requires a separately running server.

## Plan of Attack

1. Add failing regression tests for invalid live commands and frontend action gating.
2. Harden the backend event contract with structured command errors, source preservation, logging, and stale-event logging.
3. Normalize frontend live state, add `canSendLiveAction`, and apply it to live buttons, cinema controls, and keyboard shortcuts.
4. Add CodeMirror last-known-good source handling and live-specific render guards.
5. Improve live graph and teaching renderers using domain snapshots and step/source state.
6. Add browser tests for the original bug and happy path.
7. Update docs, run full QC, manually verify in browser, then commit and push.

## Completion Notes

- Root cause: live-mode cinema Play mapped directly to `continue_live` before a session existed, and LiveView patches were allowed to morph the CodeMirror-owned DOM. The invalid command produced a generic error event, then the patch detached the editor content.
- Fix: bottom Play now starts live execution when no session exists; all live actions pass through `canSendLiveAction/2`-equivalent frontend gating, and the server returns structured `command_error` events for invalid direct commands.
- CodeMirror and graph-owned DOM now use LiveView isolation, and the hook preserves the last-known-good live source across command errors and malformed events.
- Live reset is an explicit event that stops the old session, clears frontend session state, preserves source, and ignores stale old-session events.
- Micrograd domain snapshots filter instrumenter temporaries and derive graph, phase, values, and gradients from real `MicrogradEx.Value` and `MicrogradEx.Gradients` bindings.
- Tidewave-specific inspection was not available through the local tool surface; LiveView assigns were inspected through tests, rendered data attributes, server logs, and browser automation instead.

## Final Verification

- `mix precommit`: passed, 54 tests.
- `mix assets.build`: passed.
- `npm test --prefix assets`: passed, 3 files / 18 tests.
- `npm run test:e2e --prefix assets`: passed, all replay lessons plus live `x_squared`.
- Manual browser verification via Chromium against `http://localhost:4000/?subject=micrograd&lesson=x_squared&mode=live`: source visible at idle, Step disabled before session, bottom Play safely starts live session, runtime pid/session visible, Step progresses `x` before `y`, `y` has data `9`, backward shows `x` gradient `6`, graph and teaching panels update, and source never blanks.
- Replay URL `http://localhost:4000/?subject=micrograd&lesson=x_squared`: source and replay timeline render.
- Frontend debug flag verified with `window.VIZ_LIVE_DEBUG = true`; logs were emitted with `[ml-viz-live]`.
