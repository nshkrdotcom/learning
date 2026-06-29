defmodule MlVizLab.Execution.Command do
  @moduledoc """
  Frontend/runtime command metadata.
  """

  @derive Jason.Encoder
  defstruct [:id, :type, :session_id, :issued_at]

  def new(type, session_id) do
    %__MODULE__{
      id: "#{type}:#{System.unique_integer([:positive, :monotonic])}",
      type: type,
      session_id: session_id,
      issued_at: DateTime.utc_now()
    }
  end
end
