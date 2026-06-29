import {describe, expect, test} from "vitest"
import {buildSourceLineIndex, parsePayload} from "../js/viz/trace_store"

describe("trace store helpers", () => {
  test("parses JSON payloads with fallback", () => {
    expect(parsePayload("{\"ok\":true}", null)).toEqual({ok: true})
    expect(parsePayload("null", {fallback: true})).toEqual({fallback: true})
    expect(parsePayload("not-json", [])).toEqual([])
  })

  test("indexes first event for each source line", () => {
    const index = buildSourceLineIndex([
      {index: 4, source: {file: "lesson.ex", line: 2}, implementation_source: {file: "value.ex", line: 10}},
      {index: 5, source: {file: "lesson.ex", line: 2}, implementation_source: {file: "value.ex", line: 11}},
    ])

    expect(index.get("lesson.ex:2")).toBe(4)
    expect(index.get("value.ex:10")).toBe(4)
    expect(index.get("value.ex:11")).toBe(5)
  })
})

