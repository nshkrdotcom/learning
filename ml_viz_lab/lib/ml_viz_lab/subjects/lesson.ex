defmodule MlVizLab.Subjects.Lesson do
  @moduledoc """
  Subject-neutral lesson/program metadata.
  """

  @derive Jason.Encoder
  defstruct [:id, :title, :level, :description, :view, :estimated_steps]

  @type t :: %__MODULE__{
          id: String.t(),
          title: String.t(),
          level: String.t(),
          description: String.t(),
          view: String.t() | nil,
          estimated_steps: non_neg_integer() | nil
        }

  def new(attrs) when is_map(attrs), do: struct!(__MODULE__, attrs)
end
