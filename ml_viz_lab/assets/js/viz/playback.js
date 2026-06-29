import {clamp} from "./format"

function initialPlayback(totalSteps, requestedStep = 0) {
  return {
    step: clampStepForLength(Number(requestedStep || 0), totalSteps),
    speed: 1,
    playing: false,
    loopPhase: false,
    follow: true,
  }
}

function clampStepForLength(step, totalSteps) {
  return clamp(Number(step || 0), 0, Math.max(totalSteps - 1, 0))
}

function clampStepForTrace(step, events) {
  return clampStepForLength(step, (events || []).length)
}

function replaceTracePlayback(state, events, requestedStep = 0) {
  return {
    ...state,
    step: clampStepForTrace(requestedStep, events),
    playing: false,
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
    case "next_phase":
      return {...state, step: phaseBoundaryStep(state.step, action.events || [], 1)}
    case "previous_phase":
      return {...state, step: phaseBoundaryStep(state.step, action.events || [], -1)}
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

function phaseBoundaryStep(step, events, direction) {
  if (!events || events.length === 0) return 0
  const current = events[clampStepForTrace(step, events)]
  if (!current) return 0

  if (direction > 0) {
    for (let index = step + 1; index < events.length; index += 1) {
      if (events[index].phase !== current.phase) return index
    }
    return events.length - 1
  }

  for (let index = step - 1; index >= 0; index -= 1) {
    if (events[index].phase !== current.phase) {
      const previousPhase = events[index].phase
      while (index > 0 && events[index - 1].phase === previousPhase) index -= 1
      return index
    }
  }
  return 0
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

export {
  clampStepForTrace,
  initialPlayback,
  nextLoopStep,
  playbackReducer,
  replaceTracePlayback,
}
