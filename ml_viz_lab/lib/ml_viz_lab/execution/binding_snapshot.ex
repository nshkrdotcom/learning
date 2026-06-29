defmodule MlVizLab.Execution.BindingSnapshot do
  @moduledoc """
  Converts runtime bindings into safe, JSON-encodable summaries.
  """

  alias MlVizLab.Execution.SafeInspect

  def from_binding(binding) when is_list(binding) do
    binding
    |> Enum.reject(fn {name, _value} -> internal_name?(name) end)
    |> Map.new(fn {name, value} -> {Atom.to_string(name), SafeInspect.value(value)} end)
  end

  defp internal_name?(:runtime), do: true

  defp internal_name?(name) do
    name = Atom.to_string(name)
    String.starts_with?(name, "__viz_") or String.starts_with?(name, "viz_value_")
  end
end
