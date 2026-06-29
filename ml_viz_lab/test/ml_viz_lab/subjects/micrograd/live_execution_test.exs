defmodule MlVizLab.Subjects.Micrograd.LiveExecutionTest do
  use ExUnit.Case, async: true

  alias MlVizLab.Subjects.Micrograd.DomainSnapshot
  alias MlVizLab.Subjects.Micrograd.LiveLesson

  test "x_squared live lesson uses raw MicrogradEx calls" do
    source = LiveLesson.source!("x_squared")

    assert source =~ "MicrogradEx.Value.new"
    assert source =~ "MicrogradEx.Value.pow"
    assert source =~ "MicrogradEx.Value.backward"
    refute source =~ "InstrumentedValue"
    refute source =~ "MlVizLab.Subjects.Micrograd.Instrumented"
  end

  test "domain snapshots derive Micrograd graph from live bindings" do
    x = MicrogradEx.Value.new(3.0, label: "x")
    first = DomainSnapshot.from_binding(x: x)

    assert first.domain == "micrograd"
    assert Enum.any?(first.values, &(&1.name == "x" and &1.data == 3.0))
    assert length(first.graph.nodes) == 1

    y = MicrogradEx.Value.pow(x, 2, label: "y = x^2")
    second = DomainSnapshot.from_binding(x: x, y: y)

    assert length(second.graph.nodes) == 2
    assert length(second.graph.edges) == 1
    assert Enum.any?(second.values, &(&1.name == "y" and &1.data == 9.0))

    gradients = MicrogradEx.Value.backward(y)
    final = DomainSnapshot.from_binding(x: x, y: y, gradients: gradients)

    assert final.gradients[x.id] == 6.0
    assert final.active_value_name == "y"
  end
end
