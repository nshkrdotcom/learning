defmodule MlVizLab.Trace.SourceRef do
  @moduledoc """
  A source location that can be highlighted by the code panel.
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

  @type t :: %__MODULE__{
          file: String.t(),
          line: pos_integer(),
          line_start: pos_integer(),
          line_end: pos_integer(),
          column_start: pos_integer() | nil,
          column_end: pos_integer() | nil,
          token: String.t() | nil,
          kind: String.t() | nil,
          title: String.t() | nil
        }

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
