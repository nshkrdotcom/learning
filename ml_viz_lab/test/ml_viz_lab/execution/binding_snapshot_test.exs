defmodule MlVizLab.Execution.BindingSnapshotTest do
  use ExUnit.Case, async: true

  alias MicrogradEx.Value
  alias MlVizLab.Execution.BindingSnapshot
  alias MlVizLab.Execution.SafeInspect

  test "primitive bindings encode to JSON-safe snapshots" do
    snapshot = BindingSnapshot.from_binding(x: 1, name: "demo", ok: true, none: nil)

    assert snapshot["x"].kind == "number"
    assert snapshot["name"].kind == "string"
    assert snapshot["ok"].kind == "boolean"
    assert snapshot["none"].kind == "nil"
    assert Jason.encode!(snapshot)
  end

  test "large lists truncate and functions/pids do not crash" do
    snapshot =
      BindingSnapshot.from_binding(items: Enum.to_list(1..100), fun: fn -> :ok end, pid: self())

    assert snapshot["items"].kind == "list"
    assert snapshot["items"].length == 100
    assert length(snapshot["items"].preview) < 100
    assert snapshot["fun"].kind == "function"
    assert snapshot["pid"].kind == "pid"
    assert Jason.encode!(snapshot)
  end

  test "generic structs summarize module and preview fields" do
    summary = SafeInspect.value(%URI{scheme: "https", host: "example.com", path: "/demo"})

    assert summary.kind == "struct"
    assert summary.module == "URI"
    assert Enum.any?(summary.fields, &(&1.key == "host"))
    assert Jason.encode!(summary)
  end

  test "MicrogradEx values are summarized as domain values without dumping the graph" do
    value = Value.new(3.0, label: "x")
    summary = SafeInspect.value(value)

    assert summary.domain == "micrograd_value"
    assert summary.module == "MicrogradEx.Value"
    assert summary.data == 3.0
    assert summary.label == "x"
    refute Map.has_key?(summary, :graph)
    assert Jason.encode!(summary)
  end
end
