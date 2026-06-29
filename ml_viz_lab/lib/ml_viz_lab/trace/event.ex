defmodule MlVizLab.Trace.Event do
  @moduledoc """
  One immutable timeline event.
  """

  @derive Jason.Encoder
  defstruct [
    :id,
    :index,
    :phase,
    :type,
    :title,
    :source,
    :implementation_source,
    :teaching,
    :concepts,
    :snapshot,
    :value,
    :gradient,
    :parameter_update,
    :metrics,
    :compression,
    :order,
    :related_nodes,
    :related_edges,
    :actions,
    :animation,
    :debug,
    :provenance
  ]

  @type t :: %__MODULE__{}

  def new(attrs) when is_map(attrs) do
    attrs =
      attrs
      |> Map.update(:source, nil, &source/1)
      |> Map.update(:implementation_source, nil, &source/1)
      |> Map.update(:teaching, nil, &teaching/1)
      |> Map.put_new(:concepts, [])
      |> Map.put_new(:related_nodes, [])
      |> Map.put_new(:related_edges, [])
      |> Map.put_new(:actions, [])
      |> Map.put_new(:compression, nil)
      |> Map.put_new(:debug, nil)
      |> Map.put_new(:provenance, %{})

    struct!(__MODULE__, attrs)
  end

  defp source(nil), do: nil
  defp source(%MlVizLab.Trace.SourceRef{} = source), do: source

  defp source(%MlVizLab.Instrumentation.SourceSpan{} = source),
    do: MlVizLab.Trace.SourceRef.new(Map.from_struct(source))

  defp source(source) when is_map(source), do: MlVizLab.Trace.SourceRef.new(source)

  defp teaching(nil), do: nil
  defp teaching(%MlVizLab.Trace.TeachingCard{} = teaching), do: teaching
  defp teaching(teaching) when is_map(teaching), do: MlVizLab.Trace.TeachingCard.new(teaching)
end
