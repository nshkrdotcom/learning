function readStepFromUrl() {
  return Number(new URLSearchParams(window.location.search).get("step") || 0)
}

function writeStepToUrl(lessonId, step) {
  const params = new URLSearchParams(window.location.search)
  params.set("lesson", lessonId)
  params.set("step", String(step))
  history.replaceState(null, "", `${window.location.pathname}?${params}`)
}

export {readStepFromUrl, writeStepToUrl}

