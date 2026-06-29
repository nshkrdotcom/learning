defmodule MlVizLab.Execution.Event do
  @moduledoc """
  Generic live execution event.
  """

  @derive Jason.Encoder
  defstruct [
    :id,
    :type,
    :session_id,
    :runtime_pid,
    :span,
    :bindings,
    :domain_snapshot,
    :value,
    :result,
    :command_id,
    :error,
    :occurred_at
  ]

  def new(attrs) when is_map(attrs) do
    struct!(
      __MODULE__,
      attrs
      |> Map.put_new(:id, "event:#{System.unique_integer([:positive, :monotonic])}")
      |> Map.put_new(:occurred_at, DateTime.utc_now())
    )
  end
end
