defmodule MlVizLab.Subjects.MicrogradTest do
  use ExUnit.Case, async: true

  alias MlVizLab.Subjects

  test "all Micrograd lessons generate non-empty traces with source-backed events" do
    for lesson <- Subjects.lessons("micrograd") do
      trace = Subjects.generate_trace("micrograd", lesson.id)
      source_by_id = Map.new(trace.sources, &{&1.id, &1})

      assert trace.lesson_id == lesson.id
      assert trace.events != []
      assert trace.final_graph.nodes != []

      for event <- trace.events do
        assert event.index >= 0
        assert source_ref_valid?(event.source, source_by_id)
        assert source_ref_valid?(event.implementation_source, source_by_id)
      end
    end
  end

  test "x squared trace shows y = 9 and dx = 6" do
    trace = Subjects.generate_trace("micrograd", "x_squared")

    assert node(trace, "y = x^2").data == 9.0
    assert final_grad(trace, "x") == 6.0
  end

  test "repeated parent trace accumulates every contribution" do
    trace = Subjects.generate_trace("micrograd", "repeated_parent")

    assert node(trace, "x * x + x + x").data == 15.0
    assert final_grad(trace, "x") == 8.0

    repeated_edge_events =
      trace.events
      |> Enum.filter(&(&1.type == "gradient_contribution"))
      |> Enum.filter(&(&1.gradient.node_id == node(trace, "x").id))

    assert length(repeated_edge_events) >= 4
  end

  test "relu lesson records stopped and passing gradients" do
    trace = Subjects.generate_trace("micrograd", "relu_gate")

    assert final_grad(trace, "negative") == 0.0
    assert final_grad(trace, "zero") == 0.0
    assert final_grad(trace, "positive") == 1.0
  end

  test "single neuron gradients match the real MicrogradEx result" do
    trace = Subjects.generate_trace("micrograd", "single_neuron")

    assert trace.view == "network"
    assert final_grad(trace, "w0") == 2.0
    assert final_grad(trace, "w1") == -3.0
    assert final_grad(trace, "b") == 1.0
  end

  test "one training step records immutable parameter updates" do
    trace = Subjects.generate_trace("micrograd", "one_training_step")
    updates = Enum.filter(trace.events, &(&1.type == "parameter_updated"))

    assert trace.view == "training"
    assert length(updates) == 2

    [weight_update, bias_update] = Enum.map(updates, & &1.parameter_update)
    assert weight_update.old_data == 0.0
    assert_in_delta weight_update.new_data, 1.2, 1.0e-12
    assert bias_update.old_data == 0.0
    assert_in_delta bias_update.new_data, 0.6, 1.0e-12
  end

  test "compressed linear training records decreasing loss" do
    trace = Subjects.generate_trace("micrograd", "linear_training")

    losses = Enum.map(trace.events, & &1.metrics.loss)
    assert List.first(losses) > List.last(losses)
  end

  defp node(trace, label) do
    Enum.find(trace.final_graph.nodes, &(&1.label == label || &1.title == label)) ||
      flunk("expected node #{inspect(label)} in #{trace.lesson_id}")
  end

  defp final_grad(trace, label) do
    id = node(trace, label).id
    final_event = List.last(trace.events)
    final_event.snapshot.gradients[id] || final_event.snapshot.gradients[to_string(id)] || 0.0
  end

  defp source_ref_valid?(%{file: file, line: line}, source_by_id) when is_integer(line) do
    case source_by_id[file] do
      nil -> false
      source -> line >= 1 and line <= length(String.split(source.source, "\n"))
    end
  end
end
