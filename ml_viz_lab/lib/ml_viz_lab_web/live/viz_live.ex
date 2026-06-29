defmodule MlVizLabWeb.VizLive do
  use MlVizLabWeb, :live_view

  alias MlVizLab.Runs
  alias MlVizLab.Subjects

  @impl true
  def mount(_params, _session, socket) do
    subject_id = Subjects.default_subject()

    {:ok,
     socket
     |> assign(:page_title, "ML Viz Lab")
     |> assign(:subject_id, subject_id)
     |> assign(:lesson_id, default_lesson_id(subject_id))
     |> assign(:run_id, nil)
     |> assign(:run_status, "idle")
     |> assign(:trace, nil)
     |> assign(:trace_json, "null")
     |> assign(:lessons_json, Jason.encode!(Subjects.lessons(subject_id)))
     |> assign(:subjects_json, Jason.encode!(Subjects.all()))}
  end

  @impl true
  def handle_params(params, _uri, socket) do
    subject_id = params["subject"] || socket.assigns[:subject_id] || Subjects.default_subject()
    lesson_id = params["lesson"] || socket.assigns[:lesson_id] || default_lesson_id(subject_id)

    {:noreply, start_trace(socket, subject_id, lesson_id)}
  end

  @impl true
  def handle_async({:trace, run_id}, {:ok, trace}, socket) do
    if socket.assigns.run_id == run_id do
      lessons = Subjects.lessons(trace.subject_id)
      subjects = Subjects.all()

      socket =
        socket
        |> assign(:trace, trace)
        |> assign(:trace_json, Jason.encode!(trace))
        |> assign(:run_status, trace.status)
        |> assign(:lessons_json, Jason.encode!(lessons))
        |> assign(:subjects_json, Jason.encode!(subjects))
        |> push_event("trace_ready", %{
          run_id: run_id,
          status: trace.status,
          trace: trace,
          lessons: lessons,
          subjects: subjects
        })

      {:noreply, socket}
    else
      {:noreply, socket}
    end
  end

  def handle_async({:trace, run_id}, {:exit, reason}, socket) do
    if socket.assigns.run_id == run_id do
      trace = Runs.generate(socket.assigns.subject_id, socket.assigns.lesson_id, run_id)
      trace = %{trace | status: "error", error: %{type: "Exit", message: inspect(reason)}}

      {:noreply,
       socket
       |> assign(:trace, trace)
       |> assign(:trace_json, Jason.encode!(trace))
       |> assign(:run_status, "error")
       |> push_event("trace_ready", %{run_id: run_id, status: "error", trace: trace})}
    else
      {:noreply, socket}
    end
  end

  @impl true
  def render(assigns) do
    ~H"""
    <div
      id="viz-lab"
      phx-hook="VizLab"
      data-trace={@trace_json}
      data-lessons={@lessons_json}
      data-subjects={@subjects_json}
      data-run-id={@run_id}
      data-run-status={@run_status}
      class="viz-app"
    >
      <main class="viz-main">
        <section class="panel code-panel">
          <header class="panel-header">
            <div>
              <p class="panel-kicker">Code</p>
              <h2 id="code-title">lesson.ex</h2>
            </div>
            <div class="segmented" role="group" aria-label="code mode">
              <button type="button" class="seg active" data-code-mode="lesson">Lesson</button>
              <button type="button" class="seg" data-code-mode="source">Source</button>
            </div>
          </header>
          <div id="source-tabs" class="source-tabs"></div>
          <div class="code-wrap">
            <div id="code-editor" class="code-editor"></div>
            <aside id="variable-inspector" class="variable-inspector"></aside>
          </div>
        </section>

        <section class="panel graph-panel">
          <header class="panel-header graph-header">
            <div>
              <p id="phase-label" class="panel-kicker">Initialization</p>
              <h2 id="graph-title">Graph</h2>
            </div>
            <div class="toolbar" role="group" aria-label="graph controls">
              <button
                type="button"
                class="icon-btn active"
                data-view-mode="execution"
                title="Execution view"
              >E</button>
              <button
                type="button"
                class="icon-btn"
                data-view-mode="structural"
                title="Structural view"
              >S</button>
              <button type="button" class="icon-btn" data-toggle-labels title="Toggle labels">Aa</button>
              <button
                type="button"
                class="icon-btn active"
                data-toggle-derivatives
                title="Toggle local derivatives"
              >d</button>
              <button
                type="button"
                class="icon-btn active"
                data-toggle-gradients
                title="Toggle gradients"
              >g</button>
              <button
                type="button"
                class="icon-btn"
                data-action="compare"
                title="Compare from this step"
              >C</button>
              <button type="button" class="icon-btn" data-reset-camera title="Reset camera">[]</button>
            </div>
          </header>
          <div class="graph-canvas-wrap">
            <canvas id="graph-canvas"></canvas>
            <div id="graph-minimap" class="graph-minimap"></div>
            <div id="selection-card" class="selection-card is-empty"></div>
          </div>
        </section>

        <section class="panel teaching-panel">
          <header class="panel-header">
            <div>
              <p class="panel-kicker">Lesson</p>
              <h2 id="lesson-title">{if @trace, do: @trace.title, else: "Loading trace"}</h2>
            </div>
          </header>
          <div id="program-selector" class="program-selector"></div>
          <div id="progress-strip" class="progress-strip"></div>
          <div id="lesson-card" class="lesson-card"></div>
          <div id="concept-drawer" class="concept-drawer"></div>
          <div id="qa-panel" class="qa-panel">
            <div class="qa-disabled">
              <span>LLM tutor</span>
              <strong>Interface reserved</strong>
              <p>
                Context is available from the current lesson, step, source span, and selected graph object.
              </p>
            </div>
          </div>
        </section>
      </main>

      <footer class="cinema-bar">
        <div id="checkpoint-row" class="checkpoint-row"></div>
        <div class="cinema-controls">
          <button type="button" class="control-btn" data-action="start" title="Start">|&lt;</button>
          <button type="button" class="control-btn" data-action="back10" title="Back 10">&lt;&lt;</button>
          <button type="button" class="control-btn" data-action="back1" title="Back 1">&lt;</button>
          <button type="button" class="control-btn primary" data-action="toggle" title="Play">Play</button>
          <button type="button" class="control-btn" data-action="forward1" title="Forward 1">&gt;</button>
          <button type="button" class="control-btn" data-action="forward10" title="Forward 10">&gt;&gt;</button>
          <button type="button" class="control-btn" data-action="end" title="End">&gt;|</button>
          <button type="button" class="control-btn" data-action="loop_phase" title="Loop phase">Loop</button>
          <button
            type="button"
            class="control-btn active"
            data-action="follow"
            title="Follow current node"
          >Follow</button>
          <label class="speed-control">
            <span>Speed</span>
            <select id="speed-select">
              <option value="0.5">0.5x</option>
              <option value="1" selected>1x</option>
              <option value="2">2x</option>
              <option value="5">5x</option>
              <option value="10">10x</option>
            </select>
          </label>
          <input id="timeline-scrubber" class="timeline-scrubber" type="range" min="0" value="0" />
          <div id="step-counter" class="step-counter">Step 1 / 1</div>
        </div>
      </footer>
    </div>
    """
  end

  defp start_trace(socket, subject_id, lesson_id) do
    run_id = Runs.new_run_id(subject_id, lesson_id)
    lessons = Subjects.lessons(subject_id)

    socket =
      socket
      |> assign(:page_title, "ML Viz Lab")
      |> assign(:subject_id, subject_id)
      |> assign(:lesson_id, lesson_id)
      |> assign(:run_id, run_id)
      |> assign(:run_status, "loading")
      |> assign(:trace, nil)
      |> assign(:trace_json, "null")
      |> assign(:lessons_json, Jason.encode!(lessons))
      |> assign(:subjects_json, Jason.encode!(Subjects.all()))

    if connected?(socket) do
      start_async(socket, {:trace, run_id}, fn -> Runs.generate(subject_id, lesson_id, run_id) end)
    else
      socket
    end
  end

  defp default_lesson_id(subject_id) do
    subject_id
    |> Subjects.lessons()
    |> List.first()
    |> Map.fetch!(:id)
  end
end
