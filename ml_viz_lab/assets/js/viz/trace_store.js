function parsePayload(value, fallback) {
  if (!value || value === "null") return fallback
  try {
    return JSON.parse(value)
  } catch (_error) {
    return fallback
  }
}

function buildSourceLineIndex(events) {
  const index = new Map()

  for (const event of events || []) {
    for (const source of [event.source, event.implementation_source]) {
      if (!source) continue
      const key = sourceRefKey(source)
      if (!index.has(key)) index.set(key, event.index)
    }
  }

  return index
}

function sourceRefKey(source) {
  if (!source) return null
  return `${source.file}:${source.line_start || source.line}`
}

export {buildSourceLineIndex, sourceRefKey, parsePayload}
