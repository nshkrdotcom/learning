defmodule MlVizLab.Trace.Edge do
  @moduledoc """
  Subject-neutral graph edge displayed by the renderer.
  """

  @derive Jason.Encoder
  defstruct [
    :id,
    :from,
    :to,
    :from_display,
    :to_display,
    :local_gradient,
    :label,
    :category,
    :contributions,
    :creation_event_index,
    :first_backward_event_index
  ]

  @type t :: %__MODULE__{}

  def new(attrs) when is_map(attrs) do
    struct!(__MODULE__, Map.merge(%{contributions: []}, attrs))
  end
end
