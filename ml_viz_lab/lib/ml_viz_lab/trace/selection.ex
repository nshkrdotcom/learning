defmodule MlVizLab.Trace.Selection do
  @moduledoc """
  Inspectable object reference for current UI selection.
  """

  @derive Jason.Encoder
  defstruct [:type, :id, :title, :event_index]

  @type t :: %__MODULE__{}
end
