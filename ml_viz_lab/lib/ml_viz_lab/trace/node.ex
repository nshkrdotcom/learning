defmodule MlVizLab.Trace.Node do
  @moduledoc """
  Subject-neutral graph node displayed by the renderer.
  """

  @derive Jason.Encoder
  defstruct [
    :id,
    :display_id,
    :label,
    :title,
    :op,
    :kind,
    :category,
    :data,
    :is_output,
    :parents,
    :children,
    :creation_event_index,
    :first_backward_event_index,
    :related_source_refs
  ]

  @type t :: %__MODULE__{}

  def new(attrs) when is_map(attrs) do
    struct!(
      __MODULE__,
      Map.merge(
        %{
          parents: [],
          children: [],
          is_output: false,
          related_source_refs: []
        },
        attrs
      )
    )
  end
end
