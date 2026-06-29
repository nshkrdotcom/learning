defmodule MlVizLab.Instrumentation.Event do
  @moduledoc """
  Semantic event emitted while an instrumented subject lesson is executing.
  """

  alias MlVizLab.Instrumentation.SourceSpan

  @derive Jason.Encoder
  defstruct [
    :id,
    :index,
    :type,
    :phase,
    :title,
    :source,
    :implementation_source,
    :payload,
    :snapshot,
    :teaching_key,
    :concepts,
    :focus,
    :provenance
  ]

  @type t :: %__MODULE__{}

  def new(type, attrs \\ %{}) when is_atom(type) and is_map(attrs) do
    attrs =
      attrs
      |> Map.put(:type, type)
      |> Map.update(:source, nil, &source/1)
      |> Map.update(:implementation_source, nil, &source/1)
      |> Map.put_new(:payload, %{})
      |> Map.put_new(:concepts, [])
      |> Map.put_new(:provenance, %{instrumented: true})

    struct!(__MODULE__, attrs)
  end

  def with_index(%__MODULE__{} = event, index) do
    %{event | index: index, id: event.id || "#{event.type}:#{index}"}
  end

  defp source(nil), do: nil
  defp source(%SourceSpan{} = span), do: span
  defp source(source) when is_map(source), do: SourceSpan.new(source)
end
