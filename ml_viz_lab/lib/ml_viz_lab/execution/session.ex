defmodule MlVizLab.Execution.Session do
  @moduledoc """
  Live execution session controller.
  """

  use GenServer

  alias MlVizLab.Execution.Command
  alias MlVizLab.Execution.Event
  alias MlVizLab.Execution.Runtime
  alias MlVizLab.Execution.State

  def child_spec(opts) do
    %{
      id: {__MODULE__, Keyword.fetch!(opts, :session_id)},
      start: {__MODULE__, :start_link, [opts]},
      restart: :temporary,
      type: :worker
    }
  end

  def start_link(opts), do: GenServer.start_link(__MODULE__, opts)

  def state(session), do: GenServer.call(session, :state)
  def step(session), do: GenServer.call(session, {:command, :step})
  def continue(session), do: GenServer.call(session, {:command, :continue})
  def stop(session), do: GenServer.call(session, {:command, :stop})

  @impl true
  def init(opts) do
    Process.flag(:trap_exit, true)

    now = DateTime.utc_now()

    state = %State{
      session_id: Keyword.fetch!(opts, :session_id),
      subject_id: Keyword.fetch!(opts, :subject_id),
      lesson_id: Keyword.fetch!(opts, :lesson_id),
      mode: :live_ast,
      status: :starting,
      owner_pid: Keyword.get(opts, :owner_pid, self()),
      runtime_pid: nil,
      current_step: 0,
      current_span: nil,
      current_bindings: %{},
      domain_snapshot: nil,
      events: [],
      source: Keyword.fetch!(opts, :source),
      started_at: now,
      updated_at: now,
      completed_at: nil,
      error: nil,
      last_command: "start"
    }

    event =
      Event.new(%{
        type: :session_started,
        session_id: state.session_id,
        bindings: %{},
        result: nil
      })

    notify(state, event, %{source: state.source})

    {:ok,
     %{state: record_event(state, event), domain_adapter: Keyword.get(opts, :domain_adapter)},
     {:continue, :start_runtime}}
  end

  @impl true
  def handle_continue(:start_runtime, %{state: state} = data) do
    {:ok, runtime_pid} =
      Runtime.start_link(
        session_id: state.session_id,
        controller_pid: self(),
        source: state.source
      )

    {:noreply,
     %{
       data
       | state: %{
           state
           | runtime_pid: runtime_pid,
             status: :running,
             updated_at: DateTime.utc_now()
         }
     }}
  end

  @impl true
  def handle_call(:state, _from, data), do: {:reply, data.state, data}

  def handle_call({:command, :stop}, _from, %{state: state} = data) do
    command = Command.new(:stop, state.session_id)
    if state.runtime_pid, do: Runtime.stop(state.runtime_pid, state.session_id, command.id)

    {:reply, :ok,
     %{data | state: %{state | last_command: "stop", updated_at: DateTime.utc_now()}}}
  end

  def handle_call({:command, command_type}, _from, %{state: %{status: :paused} = state} = data)
      when command_type in [:step, :continue] do
    command = Command.new(command_type, state.session_id)
    Runtime.continue(state.runtime_pid, state.session_id, command.id)

    {:reply, :ok,
     %{
       data
       | state: %{
           state
           | status: :running,
             last_command: Atom.to_string(command_type),
             updated_at: DateTime.utc_now()
         }
     }}
  end

  def handle_call({:command, _command_type}, _from, data),
    do: {:reply, {:error, :not_paused}, data}

  @impl true
  def handle_info({:execution_event, event}, %{state: state} = data) do
    if event.session_id == state.session_id do
      {state, event} = apply_runtime_event(state, event, data.domain_adapter)
      notify(state, event)
      {:noreply, %{data | state: state}}
    else
      {:noreply, data}
    end
  end

  def handle_info(
        {:EXIT, pid, reason},
        %{state: %{runtime_pid: pid, status: status} = state} = data
      )
      when status not in [:completed, :error] and reason not in [:normal, :shutdown] do
    event = error_event(state, reason)

    state =
      state |> Map.put(:status, :error) |> Map.put(:error, event.error) |> record_event(event)

    notify(state, event)
    {:noreply, %{data | state: state}}
  end

  def handle_info(_message, data), do: {:noreply, data}

  defp apply_runtime_event(state, %{type: :runtime_started} = event, _domain_adapter) do
    event = Event.new(event)

    state =
      state
      |> Map.put(:runtime_pid, event.runtime_pid)
      |> Map.put(:status, :running)
      |> Map.put(:updated_at, DateTime.utc_now())
      |> record_event(event)

    {state, event}
  end

  defp apply_runtime_event(state, %{type: :paused} = event, domain_adapter) do
    domain_snapshot = domain_snapshot(domain_adapter, event.raw_binding)

    event =
      event
      |> Map.put(:domain_snapshot, domain_snapshot)
      |> Map.delete(:raw_binding)
      |> Event.new()

    state =
      state
      |> Map.put(:status, :paused)
      |> Map.put(:current_span, event.span)
      |> Map.put(:current_bindings, event.bindings)
      |> Map.put(:domain_snapshot, domain_snapshot)
      |> Map.put(:updated_at, DateTime.utc_now())
      |> record_event(event)

    {state, event}
  end

  defp apply_runtime_event(state, %{type: :binding_snapshot} = event, domain_adapter) do
    domain_snapshot = domain_snapshot(domain_adapter, event.raw_binding)

    event =
      event
      |> Map.put(:domain_snapshot, domain_snapshot)
      |> Map.delete(:raw_binding)
      |> Event.new()

    state =
      state
      |> Map.update!(:current_step, &(&1 + 1))
      |> Map.put(:current_bindings, event.bindings)
      |> Map.put(:current_span, event.span)
      |> Map.put(:domain_snapshot, domain_snapshot)
      |> Map.put(:updated_at, DateTime.utc_now())
      |> record_event(event)

    {state, event}
  end

  defp apply_runtime_event(state, %{type: :stopped} = event, _domain_adapter) do
    event = Event.new(event)

    state =
      state
      |> Map.put(:status, :stopped)
      |> Map.put(:completed_at, DateTime.utc_now())
      |> Map.put(:updated_at, DateTime.utc_now())
      |> record_event(event)

    {state, event}
  end

  defp apply_runtime_event(state, %{type: :completed} = event, domain_adapter) do
    domain_snapshot = domain_snapshot(domain_adapter, event.raw_binding)

    event =
      event
      |> Map.put(:domain_snapshot, domain_snapshot)
      |> Map.delete(:raw_binding)
      |> Event.new()

    state =
      state
      |> Map.put(:status, :completed)
      |> Map.put(:current_bindings, event.bindings || state.current_bindings)
      |> Map.put(:domain_snapshot, domain_snapshot || state.domain_snapshot)
      |> Map.put(:completed_at, DateTime.utc_now())
      |> Map.put(:updated_at, DateTime.utc_now())
      |> record_event(event)

    {state, event}
  end

  defp apply_runtime_event(state, %{type: :error} = event, _domain_adapter) do
    event = Event.new(event)

    state =
      state
      |> Map.put(:status, :error)
      |> Map.put(:error, event.error)
      |> Map.put(:completed_at, DateTime.utc_now())
      |> Map.put(:updated_at, DateTime.utc_now())
      |> record_event(event)

    {state, event}
  end

  defp apply_runtime_event(state, event, _domain_adapter) do
    event = Event.new(event)
    {record_event(%{state | updated_at: DateTime.utc_now()}, event), event}
  end

  defp domain_snapshot(nil, _binding), do: nil
  defp domain_snapshot(_domain_adapter, nil), do: nil
  defp domain_snapshot(domain_adapter, binding), do: domain_adapter.from_binding(binding)

  defp notify(%State{} = state, %Event{} = event, extra \\ %{}) do
    payload =
      event
      |> Map.from_struct()
      |> Map.merge(extra)
      |> Map.put(:status, state.status)
      |> Map.put(:current_step, state.current_step)
      |> Map.put(:session_id, state.session_id)
      |> Map.put(:subject_id, state.subject_id)
      |> Map.put(:lesson_id, state.lesson_id)
      |> Map.put(:mode, "live_ast")
      |> Map.put(:runtime_pid, pid_string(event.runtime_pid || state.runtime_pid))

    send(state.owner_pid, {:execution_event, payload})
  end

  defp record_event(%State{} = state, %Event{} = event),
    do: %{state | events: state.events ++ [event]}

  defp error_event(state, reason) do
    Event.new(%{
      type: :error,
      session_id: state.session_id,
      runtime_pid: state.runtime_pid,
      error: %{type: "RuntimeExit", message: inspect(reason)}
    })
  end

  defp pid_string(nil), do: nil
  defp pid_string(pid) when is_pid(pid), do: inspect(pid)
  defp pid_string(value), do: to_string(value)
end
