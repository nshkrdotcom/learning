defmodule MlVizLab.SubjectsTest do
  use ExUnit.Case, async: false

  alias MlVizLab.Subjects

  setup do
    original = Application.get_env(:ml_viz_lab, :subjects)

    Application.put_env(:ml_viz_lab, :subjects, [
      %{id: "ready", module: MlVizLab.TestSubjects.Ready, enabled: true, default: true},
      %{id: "disabled", module: MlVizLab.TestSubjects.Disabled, enabled: false}
    ])

    on_exit(fn -> Application.put_env(:ml_viz_lab, :subjects, original) end)
  end

  test "all returns configured enabled subjects with generic metadata" do
    assert [
             %{
               id: "ready",
               title: "Ready Subject",
               description: "A test subject that returns a ready trace.",
               lessons: [%{id: "ok"}],
               capabilities: %{testing: true}
             }
           ] = Subjects.all()
  end

  test "default_subject comes from config and disabled subjects are hidden" do
    assert Subjects.default_subject() == "ready"
    assert {:error, %{type: "UnknownSubject"}} = Subjects.get("disabled")
  end

  test "unknown subject returns a structured error" do
    assert {:error, %{type: "UnknownSubject", message: message}} = Subjects.get("missing")
    assert message =~ "missing"
  end

  test "facade exposes adapter catalogs and runs through config" do
    assert [%{id: "ok"}] = Subjects.lessons("ready")
    assert [%{id: "concept"}] = Subjects.concepts("ready")
    assert [%{id: "lesson.ex"}] = Subjects.sources("ready", "ok")

    assert {:ok, trace} = Subjects.run("ready", "ok", run_id: "ready:ok:1")
    assert trace.run_id == "ready:ok:1"
    assert trace.subject_id == "ready"
  end

  test "runs and live view orchestration do not reference Micrograd directly" do
    root = File.cwd!()

    runs_source = File.read!(Path.join(root, "lib/ml_viz_lab/runs.ex"))
    live_source = File.read!(Path.join(root, "lib/ml_viz_lab_web/live/viz_live.ex"))

    refute runs_source =~ "Subjects.Micrograd"
    refute live_source =~ "Subjects.Micrograd"
  end
end
