defmodule MlVizLab.Trace.Checkpoint do
  @moduledoc """
  Named step for phase and milestone navigation.
  """

  @derive Jason.Encoder
  defstruct [:id, :label, :step, :phase]

  @type t :: %__MODULE__{
          id: String.t(),
          label: String.t(),
          step: non_neg_integer(),
          phase: String.t() | nil
        }

  def new(attrs) when is_map(attrs), do: struct!(__MODULE__, attrs)
end
