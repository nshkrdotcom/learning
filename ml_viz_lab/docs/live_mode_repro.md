# Live Mode Repro

## Run

```sh
cd ~/p/g/n/learning/ml_viz_lab
mix phx.server
```

Open:

```text
http://localhost:4000/?subject=micrograd&lesson=x_squared&mode=live
```

## Manual Verification

- Source is visible immediately and includes `MicrogradEx.Value.new`.
- Status shows `idle`; session id and runtime pid show `-`.
- `Step`, `Continue`, and `Stop` are disabled before a session exists.
- Bottom Play before `Start live` starts live execution safely.
- Status becomes `paused`.
- Session id and runtime pid are visible.
- Source span highlights line 1.
- Step once: binding `x` appears and `y` does not.
- Step again: binding `y` appears and graph shows `x -> y`.
- Step again: gradients appear; `x` gradient is `6`.
- Source remains visible across command errors, reset, and completion.

## Debug Logs

Server logs include entries like:

```text
[live] setup_live subject=micrograd lesson=x_squared
[live] start_live session_id=... lesson=x_squared
[live] command command=step session_id=... pid=... status="paused"
[live] runtime_event type=paused session_id=... runtime_pid=...
[live] command_error command=continue reason=:no_live_session session_id=nil
[live] stale_event ignored event_session=... current_session=...
```

Frontend debug logs are disabled by default. Enable them in the browser console:

```js
window.VIZ_LIVE_DEBUG = true
```

Debug messages are prefixed with `[ml-viz-live]`.

## Tidewave / Runtime Inspection

If Tidewave or another LiveView inspection tool is available in the local environment, inspect the running `MlVizLabWeb.VizLive` assigns:

- `execution_mode`
- `live_session_id`
- `live_session_pid`
- `live_runtime_pid`
- `live_status`
- `live_source_json`
- `live_state_json`
- `live_current_span`
- `live_bindings`
- `live_domain_snapshot`
- `live_generation`

The source of truth for live progress is still backend runtime events, not the frontend inspector.
