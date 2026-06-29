defmodule MlVizLabWeb.VizLive do
  use MlVizLabWeb, :live_view

  alias MlVizLab.Execution.Controller
  alias MlVizLab.Execution.Session
  alias MlVizLab.Runs
  alias MlVizLab.Subjects

  require Logger

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
     |> assign(:live_runtime_pid, nil)
     |> assign(:live_status, "idle")
     |> assign(:live_current_span, nil)
     |> assign(:live_step_index, 0)
     |> assign(:live_source, nil)
     |> assign(:live_bindings, %{})
     |> assign(:live_domain_snapshot, nil)
     |> assign(:live_events, [])
     |> assign(:live_error, nil)
     |> assign(:live_generation, 0)
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

    Logger.info("[live] start_live session_id=#{session_id} lesson=#{lesson_id}")

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
          source: source_payload(source),
          subject_id: subject_id,
          lesson_id: lesson_id,
          mode: "live_ast"
        })

      {:noreply,
       socket
       |> assign(:live_session_id, session_id)
       |> assign(:live_session_pid, session_pid)
       |> assign(:live_runtime_pid, nil)
       |> assign(:live_status, "starting")
       |> assign(:live_current_span, nil)
       |> assign(:live_step_index, 0)
       |> assign(:live_bindings, %{})
       |> assign(:live_domain_snapshot, nil)
       |> assign(:live_error, nil)
       |> assign(:live_events, [])
       |> assign(:live_state_json, Jason.encode!(event))
       |> push_event("execution_event", event)}
    else
      {:error, error} ->
        Logger.warning(
          "[live] command_error command=start reason=#{inspect(error)} session_id=#{inspect(session_id)}"
        )

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
         |> assign(:live_error, event["error"])
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
    if socket.assigns.live_session_pid do
      _ = Session.stop(socket.assigns.live_session_pid)
    end

    socket = setup_live(socket, socket.assigns.subject_id, socket.assigns.lesson_id)

    event =
      normalize_execution_event(%{
        type: "reset",
        status: "idle",
        session_id: nil,
        subject_id: socket.assigns.subject_id,
        lesson_id: socket.assigns.lesson_id,
        source: socket.assigns.live_source,
        generation: socket.assigns.live_generation
      })

    {:noreply,
     socket
     |> assign(:live_state_json, Jason.encode!(event))
     |> push_event("execution_event", event)}
  end

  @impl true
  def handle_info({:execution_event, event}, socket) do
    if event[:session_id] == socket.assigns.live_session_id do
      normalized = normalize_execution_event(event)

      Logger.info(
        "[live] runtime_event type=#{normalized["type"]} session_id=#{normalized["session_id"]} runtime_pid=#{inspect(normalized["runtime_pid"])}"
      )

      {:noreply,
       socket
       |> apply_live_event_assigns(normalized)
       |> assign(:live_state_json, Jason.encode!(normalized))
       |> push_event("execution_event", normalized)}
    else
      Logger.warning(
        "[live] stale_event ignored event_session=#{inspect(event[:session_id])} current_session=#{inspect(socket.assigns.live_session_id)}"
      )

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
          <div class="code-wrap" data-testid="code-panel">
            <div
              id="code-editor"
              class="code-editor"
              data-testid="code-editor"
              phx-update="ignore"
            >
            </div>
            <aside id="variable-inspector" class="variable-inspector" data-testid="bindings-panel">
            </aside>
          </div>
        </section>

        <section class="panel graph-panel" data-testid="graph-panel">
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
          <div class="graph-canvas-wrap" id="graph-canvas-wrap" phx-update="ignore">
            <canvas id="graph-canvas"></canvas>
            <div id="graph-minimap" class="graph-minimap" data-testid="graph-summary"></div>
            <div id="selection-card" class="selection-card is-empty"></div>
          </div>
        </section>

        <section class="panel teaching-panel" data-testid="teaching-panel">
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
                <dt>status</dt><dd id="live-status" data-testid="live-status">idle</dd>
              </div>
              <div>
                <dt>session</dt><dd id="live-session" data-testid="session-id">-</dd>
              </div>
              <div>
                <dt>pid</dt><dd id="live-pid" data-testid="runtime-pid">-</dd>
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
            <div id="live-error" class="live-error" data-testid="live-error" hidden></div>
            <div class="live-controls">
              <button
                type="button"
                data-live-action="start"
                data-testid="live-start"
                aria-label="Start live execution"
              >Start live</button>
              <button
                type="button"
                data-live-action="step"
                data-testid="live-step"
                aria-label="Step live execution"
              >Next</button>
              <button
                type="button"
                data-live-action="continue"
                data-testid="live-continue"
                aria-label="Continue live execution"
              >Continue</button>
              <button
                type="button"
                data-live-action="stop"
                data-testid="live-stop"
                aria-label="Stop live execution"
              >Stop</button>
              <button
                type="button"
                data-live-action="reset"
                aria-label="Reset live execution"
              >Reset</button>
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
          <button
            type="button"
            class="control-btn primary"
            data-action="toggle"
            data-testid="cinema-play"
            title="Play"
            aria-label="Play or continue"
          >Play</button>
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
      |> assign(:live_runtime_pid, nil)
      |> assign(:live_status, "idle")
      |> assign(:live_current_span, nil)
      |> assign(:live_step_index, 0)
      |> assign(:live_source, nil)
      |> assign(:live_bindings, %{})
      |> assign(:live_domain_snapshot, nil)
      |> assign(:live_events, [])
      |> assign(:live_error, nil)
      |> assign(:live_generation, socket.assigns[:live_generation] || 0)
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
    Logger.info("[live] setup_live subject=#{subject_id} lesson=#{lesson_id}")

    lessons = Subjects.lessons(subject_id)

    source =
      case live_source(subject_id, lesson_id) do
        {:ok, source} -> source
        {:error, _error} -> ""
      end

    source_payload = source_payload(source)
    generation = (socket.assigns[:live_generation] || 0) + 1

    socket
    |> assign(:page_title, "ML Viz Lab")
    |> assign(:subject_id, subject_id)
    |> assign(:lesson_id, lesson_id)
    |> assign(:execution_mode, "live")
    |> assign(:live_session_id, nil)
    |> assign(:live_session_pid, nil)
    |> assign(:live_runtime_pid, nil)
    |> assign(:live_status, "idle")
    |> assign(:live_current_span, nil)
    |> assign(:live_step_index, 0)
    |> assign(:live_source, source_payload)
    |> assign(:live_bindings, %{})
    |> assign(:live_domain_snapshot, nil)
    |> assign(:live_events, [])
    |> assign(:live_error, nil)
    |> assign(:live_generation, generation)
    |> assign(:live_state_json, "null")
    |> assign(:live_source_json, Jason.encode!(source_payload))
    |> assign(:run_id, nil)
    |> assign(:run_status, "idle")
    |> assign(:trace, nil)
    |> assign(:trace_json, "null")
    |> assign(:lessons_json, Jason.encode!(lessons))
    |> assign(:subjects_json, Jason.encode!(Subjects.all()))
  end

  defp command_live(socket, command) do
    session = socket.assigns.live_session_pid

    Logger.info(
      "[live] command command=#{command} session_id=#{inspect(socket.assigns.live_session_id)} pid=#{inspect(session)} status=#{inspect(socket.assigns.live_status)}"
    )

    result =
      cond do
        is_nil(session) -> {:error, :no_live_session}
        command == :step -> Session.step(session)
        command == :continue -> Session.continue(session)
        command == :stop -> Session.stop(session)
      end

    event =
      case result do
        :ok ->
          %{
            type: "command_sent",
            status: "command_sent",
            command: Atom.to_string(command),
            session_id: socket.assigns.live_session_id,
            subject_id: socket.assigns.subject_id,
            lesson_id: socket.assigns.lesson_id,
            error: nil
          }

        {:error, reason} ->
          Logger.warning(
            "[live] command_error command=#{command} reason=#{inspect(reason)} session_id=#{inspect(socket.assigns.live_session_id)}"
          )

          command_error_event(socket, command, reason)
      end

    event = normalize_execution_event(event)

    socket =
      if event["type"] == "command_error" do
        socket
        |> assign(:live_error, event["error"])
        |> assign(:live_status, event["status"] || socket.assigns.live_status)
      else
        socket
      end

    socket
    |> assign(:live_state_json, Jason.encode!(event))
    |> push_event("execution_event", event)
  end

  defp command_error_event(socket, command, reason) do
    message =
      case {command, reason} do
        {:continue, :no_live_session} -> "Start live execution before continuing."
        {:step, :no_live_session} -> "Start live execution before stepping."
        {:stop, :no_live_session} -> "Start live execution before stopping."
        {_command, :not_paused} -> "Live execution must be paused before this command."
        {_command, reason} -> "Live command failed: #{inspect(reason)}"
      end

    %{
      type: "command_error",
      status: socket.assigns.live_status || "idle",
      command: Atom.to_string(command),
      reason: Atom.to_string(reason),
      message: message,
      session_id: socket.assigns.live_session_id,
      subject_id: socket.assigns.subject_id,
      lesson_id: socket.assigns.lesson_id,
      error: %{type: "CommandError", message: message}
    }
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

  defp apply_live_event_assigns(socket, normalized) do
    socket
    |> assign(:live_status, normalized["status"] || normalized["type"])
    |> assign(:live_runtime_pid, normalized["runtime_pid"] || socket.assigns.live_runtime_pid)
    |> assign(:live_current_span, normalized["span"] || socket.assigns.live_current_span)
    |> assign(:live_step_index, normalized["current_step"] || socket.assigns.live_step_index)
    |> assign(:live_bindings, normalized["bindings"] || socket.assigns.live_bindings)
    |> assign(
      :live_domain_snapshot,
      normalized["domain_snapshot"] || socket.assigns.live_domain_snapshot
    )
    |> assign(:live_error, normalized["error"])
    |> assign(:live_events, [normalized | socket.assigns.live_events] |> Enum.take(200))
  end

  defp source_payload(source) do
    %{
      id: "lesson.ex",
      file: "lesson.ex",
      title: "lesson.ex",
      source: source || "",
      language: "elixir"
    }
  end

  defp stringify_execution_values(nil), do: nil
  defp stringify_execution_values(value) when is_boolean(value), do: value
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
