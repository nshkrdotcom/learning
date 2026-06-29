import {clamp} from "./format"

function initialPlayback(totalSteps, requestedStep = 0) {
  return {
    step: clamp(Number(requestedStep || 0), 0, Math.max(totalSteps - 1, 0)),
    speed: 1,
    playing: false,
    loopPhase: false,
    follow: true,
  }
}

function playbackReducer(state, action, totalSteps, checkpoints = []) {
  const max = Math.max(totalSteps - 1, 0)

  switch (action.type) {
    case "start":
      return {...state, step: 0}
    case "end":
      return {...state, step: max, playing: false}
    case "step":
      return {...state, step: clamp(state.step + action.delta, 0, max)}
    case "jump":
      return {...state, step: clamp(action.step, 0, max)}
    case "checkpoint": {
      const checkpoint = checkpoints.find(item => item.id === action.id)
      return checkpoint ? {...state, step: clamp(checkpoint.step, 0, max)} : state
    }
    case "toggle":
      return {...state, playing: !state.playing}
    case "play":
      return {...state, playing: true}
    case "pause":
      return {...state, playing: false}
    case "speed":
      return {...state, speed: Number(action.speed) || 1}
    case "loop_phase":
      return {...state, loopPhase: !state.loopPhase}
    case "follow":
      return {...state, follow: !state.follow}
    default:
      return state
  }
}

function nextLoopStep(state, events) {
  const event = events[state.step]
  if (!state.loopPhase || !event) return state.step + 1

  const next = state.step + 1
  if (!events[next] || events[next].phase !== event.phase) {
    const firstInPhase = events.findIndex(candidate => candidate.phase === event.phase)
    return firstInPhase >= 0 ? firstInPhase : state.step
  }

  return next
}

export {initialPlayback, nextLoopStep, playbackReducer}

