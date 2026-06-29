import {describe, expect, test} from "vitest"
import {initialPlayback, nextLoopStep, playbackReducer} from "../js/viz/playback"

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
})

