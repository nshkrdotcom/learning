defmodule MlVizLab.Trace.TeachingCard do
  @moduledoc """
  Layered explanation text for one trace event.
  """

  @derive Jason.Encoder
  defstruct [:intuition, :mechanism, :math, :elixir, :advanced]

  @type t :: %__MODULE__{
          intuition: String.t() | nil,
          mechanism: String.t() | nil,
          math: String.t() | nil,
          elixir: String.t() | nil,
          advanced: String.t() | nil
        }

  def new(attrs) when is_map(attrs) do
    struct!(__MODULE__, attrs)
  end
end
