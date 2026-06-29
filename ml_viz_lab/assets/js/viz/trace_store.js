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
      const key = `${source.file}:${source.line}`
      if (!index.has(key)) index.set(key, event.index)
    }
  }

  return index
}

export {buildSourceLineIndex, parsePayload}

