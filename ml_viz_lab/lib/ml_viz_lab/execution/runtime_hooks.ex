defmodule MlVizLab.Execution.RuntimeHooks do
  @moduledoc """
  Hook functions injected into configured lesson ASTs.
  """

  alias MlVizLab.Execution.BindingSnapshot
  alias MlVizLab.Execution.SafeInspect

  def pause(runtime, span, binding) do
    send(runtime.controller_pid, {
      :execution_event,
      %{
        type: :paused,
        session_id: runtime.session_id,
        runtime_pid: self(),
        span: span,
        bindings: BindingSnapshot.from_binding(binding),
        raw_binding: binding
      }
    })

    receive do
      {:continue, session_id, command_id} when session_id == runtime.session_id ->
        send(runtime.controller_pid, {
          :execution_event,
          %{
            type: :resumed,
            session_id: runtime.session_id,
            runtime_pid: self(),
            command_id: command_id
          }
        })

        :ok

      {:stop, session_id, command_id} when session_id == runtime.session_id ->
        send(runtime.controller_pid, {
          :execution_event,
          %{
            type: :stopped,
            session_id: runtime.session_id,
            runtime_pid: self(),
            command_id: command_id
          }
        })

        exit(:normal)
    after
      runtime.pause_timeout_ms ->
        send(runtime.controller_pid, {
          :execution_event,
          %{
            type: :error,
            session_id: runtime.session_id,
            runtime_pid: self(),
            error: %{type: "PauseTimeout", message: "runtime pause timed out"}
          }
        })

        exit(:normal)
    end
  end

  def capture(runtime, span, binding, value \\ nil) do
    send(runtime.controller_pid, {
      :execution_event,
      %{
        type: :binding_snapshot,
        session_id: runtime.session_id,
        runtime_pid: self(),
        span: span,
        bindings: BindingSnapshot.from_binding(binding),
        raw_binding: binding,
        value: SafeInspect.value(value)
      }
    })

    :ok
  end
end
