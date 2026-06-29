function initialLiveState(source = null) {
  return {
    mode: "live_ast",
    status: "idle",
    sessionId: null,
    runtimePid: null,
    currentStep: 0,
    stepIndex: 0,
    span: null,
    currentSpan: null,
    bindings: {},
    domainSnapshot: null,
    source: normalizeSource(source),
    events: [],
    lastCommand: null,
    error: null,
    replayReady: false,
    generation: 0,
  }
}

function reduceExecutionEvent(state, event) {
  if (event?.session_id && state.sessionId && event.session_id !== state.sessionId) {
    return state
  }

  if (event?.type === "reset") {
    const reset = initialLiveState(normalizeSource(event.source) || state.source)
    return {
      ...reset,
      status: event.status || "idle",
      lastCommand: "reset",
      events: [...state.events, event],
      generation: event.generation ?? state.generation + 1,
    }
  }

  const source = normalizeSource(event?.source) || state.source
  const currentStep = event?.current_step ?? event?.step_index ?? state.currentStep
  const span = event?.span || event?.current_span || state.span
  const next = {
    ...state,
    status: event?.status || event?.type || state.status,
    sessionId: event?.session_id || state.sessionId,
    runtimePid: event?.runtime_pid || state.runtimePid,
    currentStep,
    stepIndex: currentStep,
    span,
    currentSpan: span,
    bindings: event?.bindings || state.bindings,
    domainSnapshot: event?.domain_snapshot || state.domainSnapshot,
    source,
    lastCommand: event?.command || event?.command_id || state.lastCommand,
    error: event?.error || event?.type === "command_error" ? event.error || {message: event.message} : state.error,
    events: event ? [...state.events, event] : state.events,
    generation: event?.generation ?? state.generation,
  }

  if (event?.type === "completed") next.replayReady = true
  return next
}

function canSendLiveAction(live, action) {
  switch (action) {
    case "start":
      return !live.sessionId || ["idle", "completed", "stopped", "error"].includes(live.status)
    case "step":
      return !!live.sessionId && live.status === "paused"
    case "continue":
      return !!live.sessionId && live.status === "paused"
    case "stop":
      return !!live.sessionId && !["completed", "stopped"].includes(live.status)
    case "reset":
      return true
    default:
      return false
  }
}

function normalizeSource(source) {
  if (!source) return null
  if (typeof source === "string") {
    return {id: "lesson.ex", file: "lesson.ex", title: "lesson.ex", source, language: "elixir"}
  }
  if (typeof source !== "object" || typeof source.source !== "string") return null

  const id = source.id || source.file || "lesson.ex"
  return {
    id,
    file: source.file || id,
    title: source.title || source.file || id,
    source: source.source,
    language: source.language || "elixir",
  }
}

function bindingRows(bindings = {}) {
  return Object.entries(bindings).map(([name, value]) => ({
    name,
    kind: value.kind,
    summary: value.summary || String(value.value ?? ""),
    value,
  }))
}

export {bindingRows, canSendLiveAction, initialLiveState, normalizeSource, reduceExecutionEvent}
