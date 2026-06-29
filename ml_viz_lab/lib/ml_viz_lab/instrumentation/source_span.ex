defmodule MlVizLab.Instrumentation.SourceSpan do
  @moduledoc """
  Verified source span used by instrumentation before it is serialized as a trace source ref.
  """

  @derive Jason.Encoder
  defstruct [
    :file,
    :line,
    :line_start,
    :line_end,
    :column_start,
    :column_end,
    :token,
    :kind,
    :title
  ]

  @type t :: %__MODULE__{}

  def new(attrs) when is_map(attrs) do
    line_start = Map.get(attrs, :line_start) || Map.get(attrs, :line) || 1
    line_end = Map.get(attrs, :line_end) || line_start

    struct!(
      __MODULE__,
      attrs
      |> Map.put(:line, Map.get(attrs, :line, line_start))
      |> Map.put(:line_start, line_start)
      |> Map.put(:line_end, line_end)
    )
  end

  def valid?(%__MODULE__{} = span, sources) when is_list(sources) do
    case Enum.find(sources, &(source_id(&1) == span.file)) do
      nil ->
        false

      source ->
        line_count = source |> source_body() |> String.split("\n") |> length()
        span.line_start >= 1 and span.line_end >= span.line_start and span.line_end <= line_count
    end
  end

  defp source_id(%{id: id}), do: id
  defp source_id(%{"id" => id}), do: id

  defp source_body(%{source: source}), do: source
  defp source_body(%{"source" => source}), do: source
end
