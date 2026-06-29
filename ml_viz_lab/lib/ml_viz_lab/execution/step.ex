defmodule MlVizLab.Execution.Step do
  @moduledoc """
  Completed live execution step.
  """

  @derive Jason.Encoder
  defstruct [:index, :span, :bindings, :domain_snapshot, :value]
end
