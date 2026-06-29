defmodule MlVizLab.Subjects.Micrograd.DomainSnapshot do
  @moduledoc """
  Derives Micrograd graph state from live execution bindings.
  """

  alias MicrogradEx.Gradients
  alias MicrogradEx.Value
  alias MicrogradEx.Value.Edge

  def from_binding(binding) when is_list(binding) do
    values =
      binding
      |> Enum.reject(fn {name, _value} -> internal_name?(name) end)
      |> Enum.filter(fn {_name, value} -> match?(%Value{}, value) end)
      |> Enum.map(fn {name, %Value{} = value} -> {Atom.to_string(name), value} end)

    gradient_result =
      binding
      |> Enum.find_value(%{}, fn
        {_name, %Gradients{} = gradients} ->
          %{output_id: gradients.output_id, values: Gradients.to_map(gradients)}

        _other ->
          nil
      end)

    gradients = Map.get(gradient_result, :values, %{})
    gradient_output_id = Map.get(gradient_result, :output_id)

    graph = graph(values, gradient_output_id)
    active = active_value(values, gradient_output_id)

    %{
      domain: "micrograd",
      phase: phase(values, gradients),
      values: Enum.map(values, fn {name, value} -> value_summary(name, value, gradients) end),
      graph: graph,
      gradients: gradients,
      active_value_name: active,
      active_node_id: active_node_id(values, gradient_output_id)
    }
  end

  defp graph(values, gradient_output_id) do
    nodes =
      values
      |> Enum.flat_map(fn {_name, value} -> Map.values(value.graph) end)
      |> Enum.uniq_by(& &1.id)
      |> Enum.sort_by(& &1.id)

    ordinals =
      nodes
      |> Enum.with_index(1)
      |> Map.new(fn {node, index} -> {node.id, "n#{index}"} end)

    edges =
      nodes
      |> Enum.flat_map(fn node ->
        node.parents
        |> Enum.with_index()
        |> Enum.map(fn {%Edge{} = edge, index} ->
          %{
            id: edge_id(edge.parent_id, node.id, index),
            from: edge.parent_id,
            to: node.id,
            from_display: Map.get(ordinals, edge.parent_id),
            to_display: Map.get(ordinals, node.id),
            local_gradient: edge.local_gradient,
            label: "d=#{number(edge.local_gradient)}",
            category: "dependency"
          }
        end)
      end)

    children_by_id =
      edges
      |> Enum.group_by(& &1.from, & &1.to)
      |> Map.new(fn {id, children} -> {id, Enum.uniq(children)} end)

    parent_ids_by_id =
      nodes
      |> Map.new(fn node -> {node.id, Enum.map(node.parents, & &1.parent_id)} end)

    %{
      nodes:
        Enum.map(nodes, fn node ->
          %{
            id: node.id,
            display_id: Map.get(ordinals, node.id),
            label: node.label,
            title: node.label || Map.get(ordinals, node.id),
            op: op_label(node.op),
            kind: if(node.op == :leaf, do: "leaf", else: "operation"),
            category: if(node.op == :leaf, do: "input", else: "operation"),
            data: node.data,
            is_output: node.id == output_id(values, gradient_output_id),
            parents: Map.get(parent_ids_by_id, node.id, []),
            children: Map.get(children_by_id, node.id, [])
          }
        end),
      edges: edges
    }
  end

  defp value_summary(name, %Value{} = value, gradients) do
    %{
      name: name,
      id: value.id,
      label: value.label,
      data: value.data,
      gradient: Map.get(gradients, value.id),
      summary: "#{name}: #{value.label || "value"} = #{number(value.data)}"
    }
  end

  defp active_value([], _gradient_output_id), do: nil

  defp active_value(values, gradient_output_id) do
    values
    |> active_pair(gradient_output_id)
    |> elem(0)
  end

  defp active_node_id([], _gradient_output_id), do: nil

  defp active_node_id(values, gradient_output_id) do
    values
    |> active_pair(gradient_output_id)
    |> elem(1)
    |> Map.fetch!(:id)
  end

  defp output_id([], _gradient_output_id), do: nil

  defp output_id(values, nil),
    do: values |> Enum.max_by(fn {_name, value} -> value.id end) |> elem(1) |> Map.fetch!(:id)

  defp output_id(_values, gradient_output_id), do: gradient_output_id

  defp active_pair(values, nil), do: Enum.max_by(values, fn {_name, value} -> value.id end)

  defp active_pair(values, gradient_output_id) do
    Enum.find(values, fn {_name, value} -> value.id == gradient_output_id end) ||
      active_pair(values, nil)
  end

  defp phase([], _gradients), do: "pre_run"
  defp phase(_values, gradients) when gradients == %{}, do: "forward"
  defp phase(_values, _gradients), do: "backward"

  defp edge_id(parent_id, child_id, index), do: "#{parent_id}->#{child_id}:#{index}"

  defp op_label(:leaf), do: "leaf"
  defp op_label(:+), do: "+"
  defp op_label(:-), do: "-"
  defp op_label(:*), do: "*"
  defp op_label(:neg), do: "neg"
  defp op_label(:relu), do: "relu"
  defp op_label({:pow, exponent}), do: "^#{number(exponent)}"
  defp op_label(op), do: inspect(op)

  defp number(value) when is_float(value), do: :erlang.float_to_binary(value, decimals: 4)
  defp number(value), do: inspect(value)

  defp internal_name?(:runtime), do: true

  defp internal_name?(name) do
    name = Atom.to_string(name)
    String.starts_with?(name, "__viz_") or String.starts_with?(name, "viz_value_")
  end
end
