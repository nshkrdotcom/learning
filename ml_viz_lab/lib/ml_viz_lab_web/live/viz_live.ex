defmodule MlVizLabWeb.VizLive do
  use MlVizLabWeb, :live_view

  alias MlVizLab.Execution.Controller
  alias MlVizLab.Execution.Session
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
     |> assign(:execution_mode, "replay")
     |> assign(:live_session_id, nil)
     |> assign(:live_session_pid, nil)
     |> assign(:live_status, "idle")
     |> assign(:live_state_json, "null")
     |> assign(:live_source_json, "null")
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

    if params["mode"] == "live" do
      {:noreply, setup_live(socket, subject_id, lesson_id)}
    else
      {:noreply, start_trace(socket, subject_id, lesson_id)}
    end
  end

  @impl true
  def handle_event("start_live", _params, socket) do
    subject_id = socket.assigns.subject_id
    lesson_id = socket.assigns.lesson_id
    session_id = Controller.new_session_id(subject_id, lesson_id)

    with {:ok, source} <- live_source(subject_id, lesson_id),
         {:ok, domain_adapter} <- live_domain_adapter(subject_id),
         {:ok, session_pid} <-
           Controller.start_session(
             session_id: session_id,
             subject_id: subject_id,
             lesson_id: lesson_id,
             source: source,
             owner_pid: self(),
             domain_adapter: domain_adapter
           ) do
      event =
        normalize_execution_event(%{
          type: "session_requested",
          session_id: session_id,
          status: "starting",
          source: source,
          subject_id: subject_id,
          lesson_id: lesson_id,
          mode: "live_ast"
        })

      {:noreply,
       socket
       |> assign(:live_session_id, session_id)
       |> assign(:live_session_pid, session_pid)
       |> assign(:live_status, "starting")
       |> assign(:live_state_json, Jason.encode!(event))
       |> push_event("execution_event", event)}
    else
      {:error, error} ->
        event =
          normalize_execution_event(%{
            type: "error",
            status: "error",
            session_id: session_id,
            subject_id: subject_id,
            lesson_id: lesson_id,
            error: normalize_error(error)
          })

        {:noreply,
         socket
         |> assign(:live_status, "error")
         |> assign(:live_state_json, Jason.encode!(event))
         |> push_event("execution_event", event)}
    end
  end

  def handle_event("step_live", _params, socket) do
    {:noreply, command_live(socket, :step)}
  end

  def handle_event("continue_live", _params, socket) do
    {:noreply, command_live(socket, :continue)}
  end

  def handle_event("stop_live", _params, socket) do
    {:noreply, command_live(socket, :stop)}
  end

  def handle_event("reset_live", _params, socket) do
    {:noreply, setup_live(socket, socket.assigns.subject_id, socket.assigns.lesson_id)}
  end

  @impl true
  def handle_info({:execution_event, event}, socket) do
    if event[:session_id] == socket.assigns.live_session_id do
      normalized = normalize_execution_event(event)

      {:noreply,
       socket
       |> assign(:live_status, normalized["status"] || normalized["type"])
       |> assign(:live_state_json, Jason.encode!(normalized))
       |> push_event("execution_event", normalized)}
    else
      {:noreply, socket}
    end
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
      data-execution-mode={@execution_mode}
      data-live-source={@live_source_json}
      data-live-state={@live_state_json}
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
          <div id="execution-status-card" class="execution-status-card">
            <div>
              <span class="status-pill">REPLAY</span>
              <strong>Recorded replay</strong>
            </div>
            <dl>
              <div>
                <dt>status</dt><dd id="live-status">idle</dd>
              </div>
              <div>
                <dt>session</dt><dd id="live-session">-</dd>
              </div>
              <div>
                <dt>pid</dt><dd id="live-pid">-</dd>
              </div>
              <div>
                <dt>span</dt><dd id="live-span">-</dd>
              </div>
              <div>
                <dt>step</dt><dd id="live-step">0</dd>
              </div>
              <div>
                <dt>last</dt><dd id="live-command">-</dd>
              </div>
            </dl>
            <div class="live-controls">
              <button type="button" data-live-action="start">Start live</button>
              <button type="button" data-live-action="step">Next</button>
              <button type="button" data-live-action="continue">Continue</button>
              <button type="button" data-live-action="stop">Stop</button>
              <button type="button" data-live-action="reset">Reset</button>
            </div>
          </div>
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
      |> assign(:execution_mode, "replay")
      |> assign(:live_session_id, nil)
      |> assign(:live_session_pid, nil)
      |> assign(:live_status, "idle")
      |> assign(:live_state_json, "null")
      |> assign(:live_source_json, "null")
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

  defp setup_live(socket, subject_id, lesson_id) do
    lessons = Subjects.lessons(subject_id)

    source =
      case live_source(subject_id, lesson_id) do
        {:ok, source} -> source
        {:error, _error} -> ""
      end

    socket
    |> assign(:page_title, "ML Viz Lab")
    |> assign(:subject_id, subject_id)
    |> assign(:lesson_id, lesson_id)
    |> assign(:execution_mode, "live")
    |> assign(:live_session_id, nil)
    |> assign(:live_session_pid, nil)
    |> assign(:live_status, "idle")
    |> assign(:live_state_json, "null")
    |> assign(
      :live_source_json,
      Jason.encode!(%{file: "lesson.ex", title: "lesson.ex", source: source})
    )
    |> assign(:run_id, nil)
    |> assign(:run_status, "idle")
    |> assign(:trace, nil)
    |> assign(:trace_json, "null")
    |> assign(:lessons_json, Jason.encode!(lessons))
    |> assign(:subjects_json, Jason.encode!(Subjects.all()))
  end

  defp command_live(socket, command) do
    session = socket.assigns.live_session_pid

    result =
      cond do
        is_nil(session) -> {:error, :no_live_session}
        command == :step -> Session.step(session)
        command == :continue -> Session.continue(session)
        command == :stop -> Session.stop(session)
      end

    status = if match?(:ok, result), do: "command_sent", else: "error"

    event = %{
      type: "command_sent",
      status: status,
      command: Atom.to_string(command),
      session_id: socket.assigns.live_session_id,
      subject_id: socket.assigns.subject_id,
      lesson_id: socket.assigns.lesson_id,
      error: if(match?(:ok, result), do: nil, else: normalize_error(result))
    }

    event = normalize_execution_event(event)

    socket
    |> assign(:live_state_json, Jason.encode!(event))
    |> push_event("execution_event", event)
  end

  defp default_lesson_id(subject_id) do
    subject_id
    |> Subjects.lessons()
    |> List.first()
    |> Map.fetch!(:id)
  end

  defp live_source(subject_id, lesson_id) do
    adapter = Subjects.get!(subject_id)

    if function_exported?(adapter, :live_source, 1) do
      {:ok, adapter.live_source(lesson_id)}
    else
      {:error, %{type: "NoLiveSource", message: "subject #{subject_id} has no live source"}}
    end
  rescue
    exception -> {:error, exception}
  end

  defp live_domain_adapter(subject_id) do
    adapter = Subjects.get!(subject_id)

    if function_exported?(adapter, :live_domain_adapter, 0) do
      {:ok, adapter.live_domain_adapter()}
    else
      {:ok, nil}
    end
  rescue
    exception -> {:error, exception}
  end

  defp normalize_execution_event(event) do
    event
    |> stringify_execution_values()
    |> Jason.encode!()
    |> Jason.decode!()
  end

  defp stringify_execution_values(%DateTime{} = value), do: DateTime.to_iso8601(value)

  defp stringify_execution_values(%_module{} = value),
    do: value |> Map.from_struct() |> stringify_execution_values()

  defp stringify_execution_values(value) when is_map(value) do
    value
    |> Enum.reject(fn {key, _value} -> key == :__struct__ end)
    |> Map.new(fn {key, nested} -> {key, stringify_execution_values(nested)} end)
  end

  defp stringify_execution_values(value) when is_list(value),
    do: Enum.map(value, &stringify_execution_values/1)

  defp stringify_execution_values(value) when is_atom(value), do: Atom.to_string(value)
  defp stringify_execution_values(value), do: value

  defp normalize_error({:error, error}), do: normalize_error(error)

  defp normalize_error(%{type: type, message: message}),
    do: %{type: to_string(type), message: message}

  defp normalize_error(%module{} = exception) when is_exception(exception),
    do: %{type: module |> Module.split() |> List.last(), message: Exception.message(exception)}

  defp normalize_error(error), do: %{type: "Error", message: inspect(error)}
end
