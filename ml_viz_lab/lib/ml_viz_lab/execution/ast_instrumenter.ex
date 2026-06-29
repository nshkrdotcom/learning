defmodule MlVizLab.Execution.AstInstrumenter do
  @moduledoc """
  Top-level expression AST instrumentation for live execution stepping.
  """

  alias MlVizLab.Execution.SourceSpan

  def instrument_source(source, opts \\ []) when is_binary(source) do
    case Code.string_to_quoted(source, columns: true, token_metadata: true) do
      {:ok, quoted} ->
        {instrumented, spans} = instrument_quoted(quoted, opts)
        {:ok, instrumented, spans}

      {:error, {_location, message, token}} ->
        {:error,
         %{
           type: "ParseError",
           message: Exception.format_file_line("", 0) <> "#{message} #{token}"
         }}
    end
  end

  def instrument_source!(source, opts \\ []) do
    case instrument_source(source, opts) do
      {:ok, quoted, _spans} -> quoted
      {:error, error} -> raise CompileError, description: error.message
    end
  end

  defp instrument_quoted({:__block__, meta, expressions}, opts) when is_list(expressions) do
    instrument_expressions(expressions, meta, opts)
  end

  defp instrument_quoted(expression, opts) do
    instrument_expressions([expression], [], opts)
  end

  defp instrument_expressions(expressions, meta, opts) do
    {instrumented, spans} =
      expressions
      |> Enum.with_index()
      |> Enum.map_reduce([], fn {expression, index}, spans ->
        span = span_for(expression, index, opts)
        {{instrument_expression(expression, span, index), span}, spans ++ [span]}
      end)

    {{:__block__, meta, Enum.map(instrumented, &elem(&1, 0))}, spans}
  end

  defp instrument_expression(expression, %SourceSpan{} = span, index) do
    value_var = Macro.var(:"viz_value_#{index}", nil)

    if special_form_without_value?(expression) do
      quote do
        MlVizLab.Execution.RuntimeHooks.pause(
          var!(runtime),
          unquote(Macro.escape(span)),
          binding()
        )

        unquote(expression)

        MlVizLab.Execution.RuntimeHooks.capture(
          var!(runtime),
          unquote(Macro.escape(span)),
          binding(),
          nil
        )

        nil
      end
    else
      quote do
        MlVizLab.Execution.RuntimeHooks.pause(
          var!(runtime),
          unquote(Macro.escape(span)),
          binding()
        )

        unquote(value_var) = unquote(expression)

        MlVizLab.Execution.RuntimeHooks.capture(
          var!(runtime),
          unquote(Macro.escape(span)),
          binding(),
          unquote(value_var)
        )

        unquote(value_var)
      end
    end
  end

  defp span_for(expression, index, opts) do
    meta = expression_meta(expression)
    line = Keyword.get(meta, :line, index + 1)

    SourceSpan.new(%{
      id: "expr-#{index}",
      file: Keyword.get(opts, :file, "lesson.ex"),
      line_start: line,
      line_end: line,
      column_start: Keyword.get(meta, :column),
      kind: "expression",
      title: "Expression #{index + 1}"
    })
  end

  defp expression_meta({_, meta, _}) when is_list(meta), do: meta
  defp expression_meta(_expression), do: []

  defp special_form_without_value?({form, _meta, _args}) when form in [:alias, :import, :require],
    do: true

  defp special_form_without_value?(_expression), do: false
end
