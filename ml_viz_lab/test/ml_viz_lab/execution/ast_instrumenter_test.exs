defmodule MlVizLab.Execution.AstInstrumenterTest do
  use ExUnit.Case, async: true

  alias MlVizLab.Execution.AstInstrumenter

  test "instrumented source pauses before each top-level expression and captures bindings" do
    source = """
    x = 1
    y = x + 2
    y
    """

    assert {:ok, quoted, spans} = AstInstrumenter.instrument_source(source, file: "lesson.ex")
    assert length(spans) == 3
    assert Enum.map(spans, & &1.line_start) == [1, 2, 3]

    assert Macro.to_string(quoted) =~ "MlVizLab.Execution.RuntimeHooks.pause"
    assert Macro.to_string(quoted) =~ "MlVizLab.Execution.RuntimeHooks.capture"
  end

  test "invalid source returns a structured parse error" do
    assert {:error, %{type: "ParseError", message: message}} =
             AstInstrumenter.instrument_source("x =", file: "lesson.ex")

    assert message =~ "syntax error"
  end
end
