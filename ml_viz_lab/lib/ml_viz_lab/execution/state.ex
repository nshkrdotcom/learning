defmodule MlVizLab.Execution.State do
  @moduledoc """
  Session state snapshot.
  """

  @derive Jason.Encoder
  defstruct [
    :session_id,
    :subject_id,
    :lesson_id,
    :mode,
    :status,
    :owner_pid,
    :runtime_pid,
    :current_step,
    :current_span,
    :current_bindings,
    :domain_snapshot,
    :events,
    :source,
    :started_at,
    :updated_at,
    :completed_at,
    :error,
    :last_command
  ]
end
