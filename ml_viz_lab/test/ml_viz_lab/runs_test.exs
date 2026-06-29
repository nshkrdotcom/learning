defmodule MlVizLab.RunsTest do
  use ExUnit.Case, async: false

  alias MlVizLab.Runs

  setup do
    original = Application.get_env(:ml_viz_lab, :subjects)

    Application.put_env(:ml_viz_lab, :subjects, [
      %{id: "ready", module: MlVizLab.TestSubjects.Ready, enabled: true, default: true},
      %{id: "failing", module: MlVizLab.TestSubjects.Failing, enabled: true}
    ])

    on_exit(fn -> Application.put_env(:ml_viz_lab, :subjects, original) end)
  end

  test "new run ids include subject, lesson, and a monotonic suffix" do
    first = Runs.new_run_id("ready", "ok")
    second = Runs.new_run_id("ready", "ok")

    assert first =~ ~r/^ready:ok:\d+$/
    assert second =~ ~r/^ready:ok:\d+$/
    assert first != second
  end

  test "successful run returns ready trace with requested run id and timing metadata" do
    trace = Runs.generate("ready", "ok", "ready:ok:fixed", trace_mode: :instrumented)

    assert trace.run_id == "ready:ok:fixed"
    assert trace.subject_id == "ready"
    assert trace.lesson_id == "ok"
    assert trace.status == "ready"
    assert %DateTime{} = trace.started_at
    assert %DateTime{} = trace.completed_at
    assert is_integer(trace.duration_ms)
    assert trace.duration_ms >= 0
  end

  test "failed run returns typed error trace with requested run id" do
    trace = Runs.generate("failing", "bad", "failing:bad:fixed")

    assert trace.run_id == "failing:bad:fixed"
    assert trace.status == "error"
    assert trace.error.type == "TestFailure"
    assert trace.error.message == "boom"
    assert trace.events == []
  end

  test "unknown subject is converted into an error trace" do
    trace = Runs.generate("missing", "bad", "missing:bad:fixed")

    assert trace.run_id == "missing:bad:fixed"
    assert trace.status == "error"
    assert trace.error.type == "UnknownSubject"
    assert trace.error.message =~ "missing"
  end
end
