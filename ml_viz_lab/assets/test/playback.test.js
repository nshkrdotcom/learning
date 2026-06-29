import {describe, expect, test} from "vitest"
import {
  clampStepForTrace,
  initialPlayback,
  nextLoopStep,
  playbackReducer,
  replaceTracePlayback,
} from "../js/viz/playback"

describe("playback state", () => {
  test("clamps requested initial steps", () => {
    expect(initialPlayback(5, 99).step).toBe(4)
    expect(initialPlayback(5, -10).step).toBe(0)
  })

  test("steps and jumps stay inside the trace range", () => {
    let state = initialPlayback(6, 2)

    state = playbackReducer(state, {type: "step", delta: 10}, 6)
    expect(state.step).toBe(5)

    state = playbackReducer(state, {type: "step", delta: -10}, 6)
    expect(state.step).toBe(0)

    state = playbackReducer(state, {type: "jump", step: 3}, 6)
    expect(state.step).toBe(3)
  })

  test("checkpoint navigation uses named checkpoints", () => {
    const state = initialPlayback(10, 0)
    const next = playbackReducer(state, {type: "checkpoint", id: "backward"}, 10, [
      {id: "forward", step: 2},
      {id: "backward", step: 7},
    ])

    expect(next.step).toBe(7)
  })

  test("loop phase wraps to first event in the same phase", () => {
    const events = [
      {phase: "forward"},
      {phase: "forward"},
      {phase: "backward"},
    ]
    const state = {...initialPlayback(events.length, 1), loopPhase: true}

    expect(nextLoopStep(state, events)).toBe(0)
  })

  test("phase navigation jumps to next and previous phase boundaries", () => {
    const events = [
      {phase: "initialization"},
      {phase: "forward"},
      {phase: "forward"},
      {phase: "backward"},
    ]
    let state = initialPlayback(events.length, 1)

    state = playbackReducer(state, {type: "next_phase", events}, events.length)
    expect(state.step).toBe(3)

    state = playbackReducer(state, {type: "previous_phase", events}, events.length)
    expect(state.step).toBe(1)
  })

  test("empty and replaced traces clamp playback safely", () => {
    expect(initialPlayback(0, 99).step).toBe(0)
    expect(clampStepForTrace(8, [])).toBe(0)
    expect(clampStepForTrace(8, [{}, {}, {}])).toBe(2)

    const state = {...initialPlayback(10, 8), playing: true}
    const replaced = replaceTracePlayback(state, [{}, {}, {}], 1)

    expect(replaced.step).toBe(1)
    expect(replaced.playing).toBe(false)
  })

  test("loop phase stays inside phase boundaries on ticks", () => {
    const events = [
      {phase: "forward"},
      {phase: "backward"},
      {phase: "backward"},
      {phase: "update"},
    ]
    const state = {...initialPlayback(events.length, 2), loopPhase: true}

    expect(nextLoopStep(state, events)).toBe(1)
  })
})
