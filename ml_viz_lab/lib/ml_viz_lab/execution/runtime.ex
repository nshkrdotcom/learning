defmodule MlVizLab.Execution.Runtime do
  @moduledoc """
  Evaluates AST-instrumented lesson source in a dedicated process.
  """

  alias MlVizLab.Execution.AstInstrumenter
  alias MlVizLab.Execution.BindingSnapshot
  alias MlVizLab.Execution.SafeInspect

  def start_link(opts) do
    Task.start_link(fn -> run(opts) end)
  end

  def continue(runtime_pid, session_id, command_id) do
    send(runtime_pid, {:continue, session_id, command_id})
    :ok
  end

  def stop(runtime_pid, session_id, command_id) do
    send(runtime_pid, {:stop, session_id, command_id})
    :ok
  end

  defp run(opts) do
    session_id = Keyword.fetch!(opts, :session_id)
    controller_pid = Keyword.fetch!(opts, :controller_pid)
    source = Keyword.fetch!(opts, :source)

    runtime = %{
      session_id: session_id,
      controller_pid: controller_pid,
      pause_timeout_ms: Keyword.get(opts, :pause_timeout_ms, 300_000)
    }

    send(controller_pid, {
      :execution_event,
      %{type: :runtime_started, session_id: session_id, runtime_pid: self()}
    })

    with {:ok, quoted, _spans} <- AstInstrumenter.instrument_source(source, file: "lesson.ex") do
      {result, binding} = Code.eval_quoted(quoted, [runtime: runtime], __ENV__)

      send(controller_pid, {
        :execution_event,
        %{
          type: :completed,
          session_id: session_id,
          runtime_pid: self(),
          result: SafeInspect.value(result),
          bindings: BindingSnapshot.from_binding(binding),
          raw_binding: binding
        }
      })
    else
      {:error, error} ->
        send(controller_pid, {
          :execution_event,
          %{type: :error, session_id: session_id, runtime_pid: self(), error: error}
        })
    end
  rescue
    exception ->
      send(Keyword.fetch!(opts, :controller_pid), {
        :execution_event,
        %{
          type: :error,
          session_id: Keyword.fetch!(opts, :session_id),
          runtime_pid: self(),
          error: %{
            type: exception.__struct__ |> Module.split() |> List.last(),
            message: Exception.message(exception)
          }
        }
      })
  end
end
