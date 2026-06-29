defmodule MlVizLab.Subjects.Micrograd.TraceBuilder do
  @moduledoc false

  alias MicrogradEx.Gradients
  alias MicrogradEx.Value
  alias MicrogradEx.Value.Edge
  alias MlVizLab.Subjects.Micrograd.Concepts
  alias MlVizLab.Subjects.Micrograd.SourceCatalog
  alias MlVizLab.Trace.Run

  def build(%{kind: :graph} = spec) do
    output = spec.output
    gradients = spec.gradients || Value.backward(output)
    graph = output.graph
    ordered_nodes = graph |> Map.values() |> Enum.sort_by(& &1.id)
    ordinals = ordinals(ordered_nodes)
    final_graph = serialize_graph(ordered_nodes, ordinals, output.id)

    {forward_events, visible_ids} =
      forward_events(spec, ordered_nodes, ordinals)

    topology_events = [topology_event(spec, output, ordinals, visible_ids)]
    backward_events = backward_events(spec, output, gradients, ordinals, visible_ids)

    update_events =
      update_events(
        spec,
        length(forward_events) + length(topology_events) + length(backward_events),
        ordinals,
        visible_ids
      )

    events =
      (forward_events ++ topology_events ++ backward_events ++ update_events)
      |> Enum.with_index()
      |> Enum.map(fn {event, index} ->
        event |> Map.put(:index, index) |> Map.put(:id, "#{spec.id}:#{index}")
      end)

    trace(spec, annotate_graph(final_graph, events), events)
  end

  def build(%{kind: :training_summary} = spec) do
    events =
      spec.epochs
      |> Enum.with_index()
      |> Enum.map(fn {epoch, index} ->
        %{
          id: "#{spec.id}:#{index}",
          index: index,
          phase: "training",
          type: "training_epoch",
          title: "Training step #{epoch.step}",
          source: SourceCatalog.lesson_source(spec.script, "Enum.reduce"),
          implementation_source: SourceCatalog.implementation_source(:update, spec.script),
          teaching: %{
            intuition:
              "The model runs a real MicrogradEx training step and records the loss, weight, and bias after the update.",
            mechanism:
              "For this compressed lesson, each event summarizes one full forward, backward, and immutable parameter update.",
            math:
              "w = #{number(epoch.weight)}, b = #{number(epoch.bias)}, loss = #{number(epoch.loss)}",
            elixir:
              "The old model is not mutated. MicrogradEx.NN.apply_gradients/3 returns the next model value."
          },
          concepts: ["loss", "sgd", "immutable-gradients"],
          snapshot: %{
            visible_nodes: ["loss-#{epoch.step}", "w-#{epoch.step}", "b-#{epoch.step}"],
            gradients: %{},
            active_node: "loss-#{epoch.step}",
            active_edge: nil
          },
          metrics: epoch,
          compression: Map.get(spec, :compression, %{mode: "epoch_summary", detailed: false}),
          provenance: provenance(spec)
        }
      end)

    graph = %{
      nodes:
        Enum.flat_map(spec.epochs, fn epoch ->
          [
            training_node("loss-#{epoch.step}", "loss #{epoch.step}", :loss, epoch.loss),
            training_node("w-#{epoch.step}", "w #{epoch.step}", :parameter, epoch.weight),
            training_node("b-#{epoch.step}", "b #{epoch.step}", :parameter, epoch.bias)
          ]
        end),
      edges:
        spec.epochs
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.flat_map(fn [a, b] ->
          [
            training_edge("loss-#{a.step}", "loss-#{b.step}", "next"),
            training_edge("w-#{a.step}", "w-#{b.step}", "update"),
            training_edge("b-#{a.step}", "b-#{b.step}", "update")
          ]
        end)
    }

    trace(spec, annotate_graph(graph, events), events)
  end

  defp trace(spec, final_graph, events) do
    Run.new(%{
      run_id: "#{spec.id}-#{System.unique_integer([:positive])}",
      subject_id: "micrograd",
      lesson_id: spec.id,
      title: spec.title,
      level: spec.level,
      description: spec.description,
      view: spec[:view] || "graph",
      sources: SourceCatalog.sources(spec.script),
      concepts: Concepts.all(),
      checkpoints: checkpoints(events),
      final_graph: final_graph,
      events: events,
      stats: %{
        nodes: length(final_graph.nodes),
        edges: length(final_graph.edges),
        steps: length(events),
        trace_mode: Map.get(spec, :trace_mode, "compatibility"),
        compressed: spec.kind == :training_summary
      }
    })
  end

  defp forward_events(
         %{instrumentation_events: instrumentation_events} = spec,
         _ordered_nodes,
         ordinals
       )
       when is_list(instrumentation_events) do
    instrumentation_events
    |> Enum.filter(&(&1.type in [:value_created, :operation_created, :nn_forward]))
    |> Enum.map_reduce([], fn instrumented_event, visible ->
      case instrumented_event.type do
        type when type in [:value_created, :operation_created] ->
          node = instrumented_event.payload.output_node
          visible = visible ++ [node.id]
          {forward_event(spec, instrumented_event, node, ordinals, visible), visible}

        :nn_forward ->
          {nn_forward_event(spec, instrumented_event, ordinals, visible), visible}
      end
    end)
  end

  defp forward_events(spec, ordered_nodes, ordinals) do
    Enum.map_reduce(ordered_nodes, [], fn node, visible ->
      visible = visible ++ [node.id]
      {forward_event(spec, node, ordinals, visible), visible}
    end)
  end

  defp forward_event(spec, instrumented_event, node, ordinals, visible_ids) do
    phase = if node.op == :leaf, do: "initialization", else: "forward"
    type = if node.op == :leaf, do: "leaf_created", else: "operation_created"

    %{
      phase: phase,
      type: type,
      title: instrumented_event.title || forward_title(node, ordinals),
      source: instrumented_event.source,
      implementation_source: instrumented_event.implementation_source,
      teaching: teaching_for_forward(node, ordinals),
      concepts: instrumented_event.concepts || concepts_for(node.op),
      related_nodes: [node.id | Enum.map(node.parents, & &1.parent_id)],
      related_edges:
        node.parents
        |> Enum.with_index()
        |> Enum.map(fn {edge, index} -> edge_id(edge.parent_id, node.id, index) end),
      actions: node_actions(node),
      animation: animation_for(type, phase),
      snapshot: %{
        visible_nodes: visible_ids,
        gradients: %{},
        active_node: node.id,
        active_edge: nil
      },
      value: %{
        node_id: node.id,
        display_id: Map.fetch!(ordinals, node.id),
        data: node.data,
        op: op_label(node.op)
      },
      provenance: provenance(spec, instrumented_event)
    }
  end

  defp forward_event(spec, node, ordinals, visible_ids) do
    phase = if node.op == :leaf, do: "initialization", else: "forward"
    type = if node.op == :leaf, do: "leaf_created", else: "operation_created"
    source = SourceCatalog.lesson_source(spec.script, node.label || op_label(node.op))

    %{
      phase: phase,
      type: type,
      title: forward_title(node, ordinals),
      source: source,
      implementation_source: SourceCatalog.implementation_source(node.op, spec.script),
      teaching: teaching_for_forward(node, ordinals),
      concepts: concepts_for(node.op),
      related_nodes: [node.id | Enum.map(node.parents, & &1.parent_id)],
      related_edges:
        node.parents
        |> Enum.with_index()
        |> Enum.map(fn {edge, index} -> edge_id(edge.parent_id, node.id, index) end),
      actions: node_actions(node),
      animation: animation_for(type, phase),
      snapshot: %{
        visible_nodes: visible_ids,
        gradients: %{},
        active_node: node.id,
        active_edge: nil
      },
      value: %{
        node_id: node.id,
        display_id: Map.fetch!(ordinals, node.id),
        data: node.data,
        op: op_label(node.op)
      },
      provenance: provenance(spec)
    }
  end

  defp nn_forward_event(spec, instrumented_event, ordinals, visible_ids) do
    output = instrumented_event.payload.output

    %{
      phase: "forward",
      type: "nn_forward",
      title: instrumented_event.title || "Run neural network forward",
      source: instrumented_event.source,
      implementation_source: instrumented_event.implementation_source,
      teaching: %{
        intuition:
          "The neural-network helper is just a composition of scalar MicrogradEx values.",
        mechanism:
          "Weights, inputs, and bias have already produced visible scalar operation nodes.",
        math: "activation = sum(weight_i * input_i) + bias",
        elixir: "NN.forward/2 delegates to Neuron, Layer, or MLP forward functions."
      },
      concepts: instrumented_event.concepts || ["parameters", "scalar-values"],
      related_nodes: [output.id],
      related_edges: [],
      actions: [%{id: "explain_forward", label: "Explain NN.forward", node_id: output.id}],
      animation: %{kind: "source_focus", node_id: output.id},
      snapshot: %{
        visible_nodes: visible_ids,
        gradients: %{},
        active_node: output.id,
        active_edge: nil
      },
      value: %{
        node_id: output.id,
        display_id: Map.fetch!(ordinals, output.id),
        data: output.data,
        op: "nn_forward"
      },
      provenance: provenance(spec, instrumented_event)
    }
  end

  defp topology_event(spec, output, ordinals, visible_ids) do
    order = Gradients.topological_ids(output)

    %{
      phase: "topology",
      type: "topological_order",
      title: "Topological order",
      source: SourceCatalog.lesson_source(spec.script, "backward"),
      implementation_source: SourceCatalog.implementation_source(:topology, spec.script),
      teaching: %{
        intuition:
          "Before gradients move backward, MicrogradEx orders the graph so every dependency is handled in a safe sequence.",
        mechanism:
          "The forward topological order lists leaves before the output; the backward pass uses the reverse order.",
        math: "This prevents using a gradient before all downstream contributions have arrived.",
        elixir:
          "Gradients.topological_ids/1 walks the immutable graph and avoids duplicate visits with a MapSet."
      },
      concepts: ["topological-order"],
      related_nodes: visible_ids,
      related_edges: [],
      actions: [%{id: "jump_backward", label: "Jump to backward pass", target: "backward"}],
      animation: %{kind: "topology_order"},
      snapshot: %{
        visible_nodes: visible_ids,
        gradients: %{},
        active_node: output.id,
        active_edge: nil
      },
      order: Enum.map(order, &Map.fetch!(ordinals, &1)),
      provenance: provenance(spec)
    }
  end

  defp backward_events(spec, output, expected_gradients, ordinals, visible_ids) do
    backward_order = output |> Gradients.topological_ids() |> Enum.reverse()

    seed_event = %{
      phase: "backward",
      type: "output_gradient_seeded",
      title: "Seed output gradient",
      source: SourceCatalog.lesson_source(spec.script, "backward"),
      implementation_source: SourceCatalog.implementation_source(:backward, spec.script),
      teaching: %{
        intuition: "Backward starts by saying the output changes one-for-one with itself.",
        mechanism:
          "The output node gets gradient 1.0, then that value will be sent through its parent edges.",
        math: "d(output) / d(output) = 1",
        elixir: "MicrogradEx stores this in a Gradients table, not inside the Value struct."
      },
      concepts: ["chain-rule", "immutable-gradients"],
      related_nodes: [output.id],
      related_edges: [],
      actions: [%{id: "explain_output_seed", label: "Why start at 1?", concept: "chain-rule"}],
      animation: %{kind: "gradient_seed"},
      snapshot: %{
        visible_nodes: visible_ids,
        gradients: %{output.id => 1.0},
        active_node: output.id,
        active_edge: nil
      },
      gradient: %{node_id: output.id, before: 0.0, contribution: 1.0, after: 1.0},
      provenance: provenance(spec)
    }

    {events, gradients} =
      Enum.reduce(backward_order, {[], %{output.id => 1.0}}, fn node_id, {events, gradients} ->
        node = Map.fetch!(output.graph, node_id)
        upstream = Map.get(gradients, node_id, 0.0)

        Enum.with_index(node.parents)
        |> Enum.reduce({events, gradients}, fn {%Edge{} = edge, edge_index},
                                               {events, gradients} ->
          contribution = upstream * edge.local_gradient
          before = Map.get(gradients, edge.parent_id, 0.0)
          after_value = before + contribution
          gradients = Map.put(gradients, edge.parent_id, after_value)
          edge_id = edge_id(edge.parent_id, node.id, edge_index)

          event = %{
            phase: "backward",
            type: "gradient_contribution",
            title: "Gradient to #{Map.fetch!(ordinals, edge.parent_id)}",
            source: SourceCatalog.lesson_source(spec.script, "backward"),
            implementation_source: SourceCatalog.implementation_source(:backward, spec.script),
            teaching:
              teaching_for_backward(
                node,
                edge,
                upstream,
                contribution,
                before,
                after_value,
                ordinals
              ),
            concepts: [
              "chain-rule",
              "local-derivatives",
              "gradient-accumulation",
              "immutable-gradients"
            ],
            related_nodes: [node.id, edge.parent_id],
            related_edges: [edge_id],
            actions: [
              %{id: "jump_child", label: "Focus source node", node_id: node.id},
              %{id: "jump_parent", label: "Focus target node", node_id: edge.parent_id}
            ],
            animation: %{kind: "gradient_pulse", edge_id: edge_id},
            snapshot: %{
              visible_nodes: visible_ids,
              gradients: gradients,
              active_node: edge.parent_id,
              active_edge: edge_id
            },
            gradient: %{
              node_id: edge.parent_id,
              from_node_id: node.id,
              edge_id: edge_id,
              upstream: upstream,
              local_gradient: edge.local_gradient,
              before: before,
              contribution: contribution,
              after: after_value
            },
            provenance: provenance(spec)
          }

          {events ++ [event], gradients}
        end)
      end)

    assert_gradients!(gradients, expected_gradients)
    [seed_event | events]
  end

  defp update_events(
         %{params_before: before, params_after: after_params, learning_rate: learning_rate} = spec,
         _offset,
         ordinals,
         visible_ids
       )
       when is_list(before) and is_list(after_params) do
    before
    |> Enum.zip(after_params)
    |> Enum.map(fn {%Value{} = old, %Value{} = new} ->
      gradient = spec.gradients |> Gradients.get(old)

      %{
        phase: "update",
        type: "parameter_updated",
        title: "Update #{old.label || Map.get(ordinals, old.id, "parameter")}",
        source: SourceCatalog.lesson_source(spec.script, "apply_gradients"),
        implementation_source: SourceCatalog.implementation_source(:update, spec.script),
        teaching: %{
          intuition: "Training nudges this parameter in the direction that lowers the loss.",
          mechanism: "MicrogradEx creates a new parameter Value instead of mutating the old one.",
          math:
            "#{number(old.data)} - #{number(learning_rate)} * #{number(gradient)} = #{number(new.data)}",
          elixir:
            "The updated model is a new immutable struct with fresh Value ids for its parameters."
        },
        concepts: ["parameters", "sgd", "immutable-gradients"],
        related_nodes: [old.id],
        related_edges: [],
        actions: [%{id: "explain_update", label: "Explain update rule", concept: "sgd"}],
        animation: %{kind: "parameter_update", node_id: old.id},
        snapshot: %{
          visible_nodes: visible_ids,
          gradients: gradients_to_map(spec.gradients),
          active_node: old.id,
          active_edge: nil
        },
        parameter_update: %{
          old_node_id: old.id,
          old_data: old.data,
          new_data: new.data,
          gradient: gradient,
          learning_rate: learning_rate
        },
        provenance: provenance(spec)
      }
    end)
  end

  defp update_events(_spec, _offset, _ordinals, _visible_ids), do: []

  defp assert_gradients!(actual, expected) do
    expected_map = Gradients.to_map(expected)

    unless close_maps?(actual, expected_map) do
      raise "trace replay gradients did not match MicrogradEx.backward/1"
    end
  end

  defp close_maps?(left, right) do
    Map.keys(left) |> Enum.sort() == Map.keys(right) |> Enum.sort() and
      Enum.all?(left, fn {key, value} -> abs(value - Map.fetch!(right, key)) < 1.0e-9 end)
  end

  defp serialize_graph(ordered_nodes, ordinals, output_id) do
    edges =
      ordered_nodes
      |> Enum.flat_map(fn node ->
        node.parents
        |> Enum.with_index()
        |> Enum.map(fn {%Edge{} = edge, index} ->
          %{
            id: edge_id(edge.parent_id, node.id, index),
            from: edge.parent_id,
            to: node.id,
            from_display: Map.fetch!(ordinals, edge.parent_id),
            to_display: Map.fetch!(ordinals, node.id),
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
      ordered_nodes
      |> Map.new(fn node -> {node.id, Enum.map(node.parents, & &1.parent_id)} end)

    nodes =
      Enum.map(ordered_nodes, fn node ->
        %{
          id: node.id,
          display_id: Map.fetch!(ordinals, node.id),
          label: node.label,
          title: node.label || Map.fetch!(ordinals, node.id),
          op: op_label(node.op),
          kind: if(node.op == :leaf, do: "leaf", else: "operation"),
          category: node_category(node),
          data: node.data,
          is_output: node.id == output_id,
          parents: Map.get(parent_ids_by_id, node.id, []),
          children: Map.get(children_by_id, node.id, [])
        }
      end)

    %{nodes: nodes, edges: edges}
  end

  defp annotate_graph(%{nodes: nodes, edges: edges} = graph, events) do
    creation_indexes =
      events
      |> Enum.filter(&(&1.type in ["leaf_created", "operation_created"]))
      |> Map.new(fn event -> {event.value.node_id, event.index} end)

    first_backward_indexes =
      events
      |> Enum.filter(&(&1.type in ["output_gradient_seeded", "gradient_contribution"]))
      |> Enum.reduce(%{}, fn event, indexes ->
        node_id = event.gradient && event.gradient.node_id
        if node_id, do: Map.put_new(indexes, node_id, event.index), else: indexes
      end)

    source_refs =
      events
      |> Enum.flat_map(fn event ->
        Enum.map(Map.get(event, :related_nodes, []), fn node_id ->
          {node_id, Enum.reject([event.source, event.implementation_source], &is_nil/1)}
        end)
      end)
      |> Enum.group_by(fn {node_id, _refs} -> node_id end, fn {_node_id, refs} -> refs end)
      |> Map.new(fn {node_id, nested_refs} ->
        refs = nested_refs |> List.flatten() |> Enum.uniq()
        {node_id, refs}
      end)

    contribution_by_edge =
      events
      |> Enum.filter(&(&1.type == "gradient_contribution"))
      |> Enum.group_by(& &1.gradient.edge_id, fn event ->
        %{
          event_index: event.index,
          upstream: event.gradient.upstream,
          local_gradient: event.gradient.local_gradient,
          contribution: event.gradient.contribution,
          before: event.gradient.before,
          after: event.gradient.after
        }
      end)

    nodes =
      Enum.map(nodes, fn node ->
        node
        |> Map.put(:creation_event_index, Map.get(creation_indexes, node.id))
        |> Map.put(:first_backward_event_index, Map.get(first_backward_indexes, node.id))
        |> Map.put(:related_source_refs, Map.get(source_refs, node.id, []))
      end)

    edges =
      Enum.map(edges, fn edge ->
        contributions = Map.get(contribution_by_edge, edge.id, [])
        first_backward = contributions |> List.first() |> then(&(&1 && &1.event_index))

        edge
        |> Map.put(:contributions, contributions)
        |> Map.put(:creation_event_index, Map.get(creation_indexes, edge.to))
        |> Map.put(:first_backward_event_index, first_backward)
      end)

    %{graph | nodes: nodes, edges: edges}
  end

  defp ordinals(ordered_nodes) do
    ordered_nodes
    |> Enum.with_index(1)
    |> Map.new(fn {node, index} -> {node.id, "n#{index}"} end)
  end

  defp checkpoints(events) do
    [
      checkpoint("start", "Start", 0),
      first_phase(events, "forward", "Forward"),
      first_phase(events, "topology", "Topo"),
      first_phase(events, "backward", "Backward"),
      first_phase(events, "update", "Update"),
      checkpoint("end", "End", max(length(events) - 1, 0))
    ]
    |> Enum.reject(&is_nil/1)
    |> Enum.uniq_by(& &1.id)
  end

  defp first_phase(events, phase, label) do
    case Enum.find_index(events, &(&1.phase == phase)) do
      nil -> nil
      index -> checkpoint(phase, label, index)
    end
  end

  defp checkpoint(id, label, index), do: %{id: id, label: label, step: index, phase: id}

  defp forward_title(%{op: :leaf} = node, ordinals),
    do: "Create #{node.label || Map.fetch!(ordinals, node.id)}"

  defp forward_title(node, _ordinals), do: "Create #{op_label(node.op)} node"

  defp teaching_for_forward(%{op: :leaf} = node, _ordinals) do
    label = node.label || "this leaf"

    %{
      intuition: "#{label} enters the computation as a scalar value.",
      mechanism: "A leaf has data but no parents, so it is a starting point in the graph.",
      math: "value = #{number(node.data)}",
      elixir: "Value.new/2 creates an immutable Value and one graph Node record."
    }
  end

  defp teaching_for_forward(node, ordinals) do
    parents = Enum.map_join(node.parents, ", ", &Map.fetch!(ordinals, &1.parent_id))

    %{
      intuition: "A new scalar result appears from the operation #{op_label(node.op)}.",
      mechanism:
        "The node stores parent edges from #{parents}; each edge remembers a local derivative.",
      math: "output data = #{number(node.data)}",
      elixir: "The operation merges parent graphs and returns a new Value handle for this output."
    }
  end

  defp teaching_for_backward(node, edge, upstream, contribution, before, after_value, ordinals) do
    child = Map.fetch!(ordinals, node.id)
    parent = Map.fetch!(ordinals, edge.parent_id)

    %{
      intuition: "Gradient flows from #{child} back into #{parent}.",
      mechanism:
        "The upstream gradient is multiplied by the edge's local derivative, then added to the parent total.",
      math:
        "#{number(upstream)} * #{number(edge.local_gradient)} = #{number(contribution)}; #{number(before)} + #{number(contribution)} = #{number(after_value)}",
      elixir:
        "Gradients.backward/1 updates a map entry for #{parent}; the Value itself is unchanged."
    }
  end

  defp concepts_for(:leaf), do: ["scalar-values"]
  defp concepts_for(:relu), do: ["local-derivatives", "relu-gate"]
  defp concepts_for(:*), do: ["local-derivatives"]
  defp concepts_for({:pow, _}), do: ["local-derivatives", "chain-rule"]
  defp concepts_for(_op), do: ["local-derivatives", "scalar-values"]

  defp training_node(id, title, op, data) do
    %{
      id: id,
      display_id: id,
      label: title,
      title: title,
      op: Atom.to_string(op),
      kind: "summary",
      category: Atom.to_string(op),
      data: data,
      is_output: op == :loss,
      parents: [],
      children: []
    }
  end

  defp training_edge(from, to, label) do
    %{
      id: "#{from}->#{to}",
      from: from,
      to: to,
      from_display: from,
      to_display: to,
      local_gradient: nil,
      label: label,
      category: "training_progress"
    }
  end

  defp gradients_to_map(%Gradients{} = gradients), do: Gradients.to_map(gradients)

  defp edge_id(parent_id, child_id, index), do: "#{parent_id}->#{child_id}:#{index}"

  defp op_label(:leaf), do: "leaf"
  defp op_label(:+), do: "+"
  defp op_label(:-), do: "-"
  defp op_label(:*), do: "*"
  defp op_label(:neg), do: "neg"
  defp op_label(:relu), do: "relu"
  defp op_label({:pow, exponent}), do: "^#{number(exponent)}"
  defp op_label(op), do: inspect(op)

  defp node_category(%{op: :leaf, label: label}) when is_binary(label) do
    cond do
      String.match?(label, ~r/^w\d*$|weight/i) -> "parameter"
      String.match?(label, ~r/^b$|bias/i) -> "bias"
      String.contains?(label, "target") -> "target"
      true -> "input"
    end
  end

  defp node_category(%{op: :leaf}), do: "input"
  defp node_category(%{op: :relu}), do: "activation"

  defp node_category(%{label: label}) when is_binary(label) do
    cond do
      String.contains?(label, "loss") -> "loss"
      String.contains?(label, "relu") -> "activation"
      true -> "operation"
    end
  end

  defp node_category(_node), do: "operation"

  defp node_actions(%{op: :leaf} = node) do
    [%{id: "explain_node", label: "Explain #{node.label || "leaf"}", node_id: node.id}]
  end

  defp node_actions(node) do
    [
      %{id: "explain_node", label: "Explain operation", node_id: node.id},
      %{id: "jump_backward", label: "Find backward step", node_id: node.id}
    ]
  end

  defp animation_for("leaf_created", _phase), do: %{kind: "materialize_node"}
  defp animation_for("operation_created", _phase), do: %{kind: "materialize_operation"}
  defp animation_for(_type, "backward"), do: %{kind: "gradient_flow"}
  defp animation_for(_type, phase), do: %{kind: phase}

  defp number(value) when is_integer(value), do: Integer.to_string(value)
  defp number(value) when is_float(value), do: :erlang.float_to_binary(value, decimals: 4)
  defp number(value), do: inspect(value)

  defp provenance(spec, event \\ nil) do
    Map.merge(
      %{
        instrumented: Map.get(spec, :trace_mode) == "instrumented",
        trace_mode: Map.get(spec, :trace_mode, "compatibility")
      },
      if(event && event.provenance, do: event.provenance, else: %{})
    )
  end
end
