import {describe, expect, test} from "vitest"
import {bindingRows, initialLiveState, reduceExecutionEvent} from "../js/viz/live_state"

describe("live execution state", () => {
  test("paused event updates status, span, and bindings", () => {
    const state = reduceExecutionEvent(initialLiveState(), {
      type: "paused",
      status: "paused",
      session_id: "s1",
      runtime_pid: "#PID<0.1.0>",
      current_step: 0,
      span: {file: "lesson.ex", line_start: 1},
      bindings: {},
    })

    expect(state.status).toBe("paused")
    expect(state.sessionId).toBe("s1")
    expect(state.span.line_start).toBe(1)
  })

  test("binding snapshots update inspector model", () => {
    const state = reduceExecutionEvent(initialLiveState(), {
      type: "binding_snapshot",
      current_step: 1,
      bindings: {x: {kind: "number", value: 1, summary: "1"}},
    })

    expect(state.currentStep).toBe(1)
    expect(bindingRows(state.bindings)).toEqual([
      {name: "x", kind: "number", summary: "1", value: {kind: "number", value: 1, summary: "1"}},
    ])
  })

  test("completed enables replay availability marker", () => {
    const state = reduceExecutionEvent(initialLiveState(), {type: "completed", status: "completed"})
    expect(state.replayReady).toBe(true)
  })
})
