function initialLiveState(source = null) {
  return {
    mode: "live_ast",
    status: "idle",
    sessionId: null,
    runtimePid: null,
    currentStep: 0,
    span: null,
    bindings: {},
    domainSnapshot: null,
    source,
    events: [],
    lastCommand: null,
    error: null,
    replayReady: false,
  }
}

function reduceExecutionEvent(state, event) {
  const next = {
    ...state,
    status: event.status || event.type || state.status,
    sessionId: event.session_id || state.sessionId,
    runtimePid: event.runtime_pid || state.runtimePid,
    currentStep: event.current_step ?? state.currentStep,
    span: event.span || state.span,
    bindings: event.bindings || state.bindings,
    domainSnapshot: event.domain_snapshot || state.domainSnapshot,
    source: event.source ? {file: "lesson.ex", title: "lesson.ex", source: event.source} : state.source,
    lastCommand: event.command || event.command_id || state.lastCommand,
    error: event.error || state.error,
    events: [...state.events, event],
  }

  if (event.type === "completed") next.replayReady = true
  return next
}

function bindingRows(bindings = {}) {
  return Object.entries(bindings).map(([name, value]) => ({
    name,
    kind: value.kind,
    summary: value.summary || String(value.value ?? ""),
    value,
  }))
}

export {bindingRows, initialLiveState, reduceExecutionEvent}
