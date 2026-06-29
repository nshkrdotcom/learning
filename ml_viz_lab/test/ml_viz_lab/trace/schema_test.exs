defmodule MlVizLab.Trace.SchemaTest do
  use ExUnit.Case, async: true

  alias MlVizLab.Trace.Event
  alias MlVizLab.Trace.Run
  alias MlVizLab.Trace.SourceRef

  test "event JSON uses stable rich trace keys" do
    event =
      Event.new(%{
        id: "event",
        index: 0,
        phase: "forward",
        type: "operation_created",
        title: "Create op",
        source: %{file: "lesson.ex", line: 2, line_start: 2, line_end: 2, token: "x"},
        implementation_source: %{file: "value.ex", line: 120, line_start: 120, line_end: 128},
        teaching: %{intuition: "real"},
        concepts: ["chain-rule"],
        related_nodes: [1],
        related_edges: ["1->2:0"],
        actions: [],
        animation: %{kind: "materialize"},
        snapshot: %{visible_nodes: [1], gradients: %{}, active_node: 1, active_edge: nil},
        value: %{node_id: 1, data: 2.0},
        gradient: nil,
        parameter_update: nil,
        metrics: nil,
        compression: nil,
        provenance: %{instrumented: true}
      })

    encoded = Jason.encode!(event)

    for key <- [
          "id",
          "index",
          "phase",
          "type",
          "source",
          "implementation_source",
          "snapshot",
          "compression",
          "provenance"
        ] do
      assert encoded =~ ~s("#{key}")
    end
  end

  test "source refs support verified line ranges while retaining line compatibility" do
    ref = SourceRef.new(%{file: "value.ex", line_start: 10, line_end: 12, token: "def add"})

    assert ref.line == 10
    assert ref.line_start == 10
    assert ref.line_end == 12
  end

  test "run schema carries status, timing, stats, and capabilities" do
    now = DateTime.utc_now()

    run =
      Run.new(%{
        run_id: "run",
        subject_id: "subject",
        lesson_id: "lesson",
        title: "Lesson",
        level: "Level",
        description: "Desc",
        view: "graph",
        sources: [],
        concepts: [],
        checkpoints: [],
        final_graph: %{nodes: [], edges: []},
        events: [],
        stats: %{nodes: 0, edges: 0, steps: 0},
        started_at: now,
        completed_at: now,
        duration_ms: 1,
        capabilities: %{source_modes: ["lesson"]}
      })

    assert run.status == "ready"
    assert run.started_at == now
    assert run.completed_at == now
    assert run.duration_ms == 1
    assert run.capabilities.source_modes == ["lesson"]
  end
end
