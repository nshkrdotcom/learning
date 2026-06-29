import {describe, expect, test} from "vitest"
import {
  bindingRows,
  canSendLiveAction,
  initialLiveState,
  reduceExecutionEvent,
} from "../js/viz/live_state"

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

  test("live actions are gated before a backend session exists", () => {
    const idle = initialLiveState({id: "lesson.ex", title: "lesson.ex", source: "x = 1"})

    expect(canSendLiveAction(idle, "start")).toBe(true)
    expect(canSendLiveAction(idle, "step")).toBe(false)
    expect(canSendLiveAction(idle, "continue")).toBe(false)
    expect(canSendLiveAction(idle, "stop")).toBe(false)
    expect(canSendLiveAction(idle, "reset")).toBe(true)
  })

  test("paused live sessions can step, continue, and stop", () => {
    const paused = {
      ...initialLiveState(),
      status: "paused",
      sessionId: "session-1",
    }

    expect(canSendLiveAction(paused, "start")).toBe(false)
    expect(canSendLiveAction(paused, "step")).toBe(true)
    expect(canSendLiveAction(paused, "continue")).toBe(true)
    expect(canSendLiveAction(paused, "stop")).toBe(true)
  })

  test("command errors preserve the previous live source", () => {
    const source = {id: "lesson.ex", title: "lesson.ex", source: "x = 1"}
    const state = reduceExecutionEvent(initialLiveState(source), {
      type: "command_error",
      status: "idle",
      command: "continue",
      reason: "no_live_session",
      error: {message: "Start live execution before continuing."},
    })

    expect(state.source).toEqual({
      file: "lesson.ex",
      id: "lesson.ex",
      language: "elixir",
      source: "x = 1",
      title: "lesson.ex",
    })
    expect(state.error.message).toBe("Start live execution before continuing.")
  })

  test("stale live events do not mutate state", () => {
    const current = {
      ...initialLiveState(),
      sessionId: "current",
      status: "paused",
      bindings: {x: {summary: "x"}},
    }

    const state = reduceExecutionEvent(current, {
      type: "binding_snapshot",
      session_id: "old",
      bindings: {y: {summary: "y"}},
    })

    expect(state).toBe(current)
  })

  test("reset clears session state and preserves source", () => {
    const source = {id: "lesson.ex", title: "lesson.ex", source: "x = 1", language: "elixir"}
    const running = {
      ...initialLiveState(source),
      sessionId: "session-1",
      runtimePid: "#PID<0.1.0>",
      status: "paused",
      currentStep: 2,
      bindings: {x: {summary: "x"}},
    }

    const state = reduceExecutionEvent(running, {
      type: "reset",
      status: "idle",
      session_id: null,
      source,
      generation: 2,
    })

    expect(state.sessionId).toBe(null)
    expect(state.runtimePid).toBe(null)
    expect(state.currentStep).toBe(0)
    expect(state.bindings).toEqual({})
    expect(state.source.source).toBe("x = 1")
    expect(state.generation).toBe(2)
  })
})
