function readStepFromUrl() {
  return Number(new URLSearchParams(window.location.search).get("step") || 0)
}

function writeStepToUrl(lessonId, step, subjectId = null) {
  const params = new URLSearchParams(window.location.search)
  if (subjectId) params.set("subject", subjectId)
  params.set("lesson", lessonId)
  params.set("step", String(step))
  history.replaceState(null, "", `${window.location.pathname}?${params}`)
}

export {readStepFromUrl, writeStepToUrl}
