defmodule MlVizLab.TestSubjects.Ready do
  @behaviour MlVizLab.Subjects.Adapter

  alias MlVizLab.Trace.Run

  @impl true
  def id, do: "ready"

  @impl true
  def title, do: "Ready Subject"

  @impl true
  def description, do: "A test subject that returns a ready trace."

  @impl true
  def capabilities, do: %{testing: true}

  @impl true
  def lessons, do: [%{id: "ok", title: "OK", level: "Test", description: "OK lesson"}]

  @impl true
  def concepts, do: [%{id: "concept", title: "Concept", body: "Test concept"}]

  @impl true
  def sources(_lesson_id),
    do: [%{id: "lesson.ex", title: "lesson.ex", path: "test", source: "ok\n"}]

  @impl true
  def run("ok", opts) do
    {:ok,
     Run.new(%{
       run_id: Keyword.fetch!(opts, :run_id),
       subject_id: id(),
       lesson_id: "ok",
       title: "OK",
       level: "Test",
       description: "OK lesson",
       view: "graph",
       sources: sources("ok"),
       concepts: concepts(),
       checkpoints: [],
       final_graph: %{nodes: [], edges: []},
       events: [],
       stats: %{nodes: 0, edges: 0, steps: 0},
       capabilities: capabilities()
     })}
  end

  def run(_lesson_id, _opts), do: {:error, %{type: "UnknownLesson", message: "missing"}}
end

defmodule MlVizLab.TestSubjects.Disabled do
  @behaviour MlVizLab.Subjects.Adapter

  @impl true
  def id, do: "disabled"

  @impl true
  def title, do: "Disabled Subject"

  @impl true
  def description, do: "A disabled test subject."

  @impl true
  def capabilities, do: %{}

  @impl true
  def lessons, do: []

  @impl true
  def concepts, do: []

  @impl true
  def sources(_lesson_id), do: []

  @impl true
  def run(_lesson_id, _opts), do: {:error, :disabled}
end

defmodule MlVizLab.TestSubjects.Failing do
  @behaviour MlVizLab.Subjects.Adapter

  @impl true
  def id, do: "failing"

  @impl true
  def title, do: "Failing Subject"

  @impl true
  def description, do: "A test subject that fails."

  @impl true
  def capabilities, do: %{}

  @impl true
  def lessons, do: [%{id: "bad", title: "Bad", level: "Test", description: "Bad lesson"}]

  @impl true
  def concepts, do: []

  @impl true
  def sources(_lesson_id), do: []

  @impl true
  def run(_lesson_id, _opts), do: {:error, %{type: "TestFailure", message: "boom"}}
end
