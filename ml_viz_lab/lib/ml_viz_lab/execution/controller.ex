defmodule MlVizLab.Execution.Controller do
  @moduledoc """
  Facade for starting live execution sessions.
  """

  alias MlVizLab.Execution.SessionSupervisor

  def new_session_id(subject_id, lesson_id) do
    "#{subject_id}:#{lesson_id}:live:#{System.unique_integer([:positive, :monotonic])}"
  end

  def start_session(opts), do: SessionSupervisor.start_session(opts)
end
