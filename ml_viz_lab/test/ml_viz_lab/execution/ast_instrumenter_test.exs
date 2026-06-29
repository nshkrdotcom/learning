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

  test "comments and blank lines do not produce expression spans" do
    source = """
    # setup x
    x = 1

    # derive y
    y = x + 2
    """

    assert {:ok, _quoted, spans} = AstInstrumenter.instrument_source(source, file: "lesson.ex")

    assert Enum.map(spans, & &1.line_start) == [2, 5]
    assert Enum.all?(spans, &(&1.file == "lesson.ex"))
  end
end
