defmodule MlVizLab.Instrumentation.Snapshot do
  @moduledoc """
  Small helpers for trace replay snapshots.
  """

  def graph(visible_nodes, gradients, active_node, active_edge \\ nil) do
    %{
      visible_nodes: visible_nodes,
      gradients: gradients || %{},
      active_node: active_node,
      active_edge: active_edge
    }
  end
end
