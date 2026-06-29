defmodule MlVizLab.Execution.SessionSupervisor do
  @moduledoc """
  Dynamic supervisor for live execution sessions.
  """

  use DynamicSupervisor

  def start_link(opts) do
    DynamicSupervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(_opts), do: DynamicSupervisor.init(strategy: :one_for_one)

  def start_session(opts) do
    DynamicSupervisor.start_child(__MODULE__, {MlVizLab.Execution.Session, opts})
  end
end
