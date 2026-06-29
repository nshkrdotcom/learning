defmodule MlVizLab.Trace.Run do
  @moduledoc """
  Complete immutable trace run sent to the browser.
  """

  @derive Jason.Encoder
  defstruct [
    :run_id,
    :subject_id,
    :lesson_id,
    :title,
    :level,
    :description,
    :view,
    :status,
    :sources,
    :concepts,
    :checkpoints,
    :final_graph,
    :events,
    :stats,
    :error,
    :started_at,
    :completed_at,
    :duration_ms,
    :capabilities
  ]

  @type t :: %__MODULE__{}

  def new(attrs) when is_map(attrs) do
    attrs =
      attrs
      |> Map.put_new(:status, "ready")
      |> Map.put_new(:sources, [])
      |> Map.put_new(:concepts, [])
      |> Map.put_new(:checkpoints, [])
      |> Map.put_new(:final_graph, %{nodes: [], edges: []})
      |> Map.put_new(:events, [])
      |> Map.put_new(:stats, %{nodes: 0, edges: 0, steps: 0})
      |> Map.put_new(:capabilities, %{})
      |> Map.update(:events, [], &Enum.map(&1, fn event -> event(event) end))
      |> Map.update(:checkpoints, [], &Enum.map(&1, fn checkpoint -> checkpoint(checkpoint) end))
      |> Map.update(:final_graph, %{nodes: [], edges: []}, &final_graph/1)

    struct!(__MODULE__, attrs)
  end

  defp event(%MlVizLab.Trace.Event{} = event), do: event
  defp event(event) when is_map(event), do: MlVizLab.Trace.Event.new(event)

  defp checkpoint(%MlVizLab.Trace.Checkpoint{} = checkpoint), do: checkpoint

  defp checkpoint(checkpoint) when is_map(checkpoint),
    do: MlVizLab.Trace.Checkpoint.new(checkpoint)

  defp final_graph(%{nodes: nodes, edges: edges} = graph) do
    %{
      graph
      | nodes: Enum.map(nodes, &trace_node/1),
        edges: Enum.map(edges, &trace_edge/1)
    }
  end

  defp trace_node(%MlVizLab.Trace.Node{} = node), do: node
  defp trace_node(node) when is_map(node), do: MlVizLab.Trace.Node.new(node)

  defp trace_edge(%MlVizLab.Trace.Edge{} = edge), do: edge
  defp trace_edge(edge) when is_map(edge), do: MlVizLab.Trace.Edge.new(edge)
end
