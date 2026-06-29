defmodule MlVizLab.Execution.SourceSpan do
  @moduledoc """
  Source span for live AST execution.
  """

  @derive Jason.Encoder
  defstruct [
    :id,
    :file,
    :line,
    :line_start,
    :line_end,
    :column_start,
    :column_end,
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
end
