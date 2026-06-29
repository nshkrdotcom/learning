defmodule MlVizLab.Execution.SafeInspect do
  @moduledoc """
  JSON-safe summaries for arbitrary live execution values.
  """

  alias MicrogradEx.Gradients
  alias MicrogradEx.Value

  @max_preview 6
  @max_string 160

  def value(value), do: value(value, 0)

  defp value(nil, _depth), do: %{kind: "nil", value: nil, summary: "nil"}

  defp value(value, _depth) when is_boolean(value),
    do: %{kind: "boolean", value: value, summary: inspect(value)}

  defp value(value, _depth) when is_number(value),
    do: %{kind: "number", value: value, summary: inspect(value)}

  defp value(value, _depth) when is_atom(value),
    do: %{kind: "atom", value: Atom.to_string(value), summary: inspect(value)}

  defp value(value, _depth) when is_binary(value) do
    %{kind: "string", value: truncate(value), summary: inspect(truncate(value))}
  end

  defp value(%Value{} = value, _depth) do
    %{
      kind: "struct",
      domain: "micrograd_value",
      module: "MicrogradEx.Value",
      id: value.id,
      data: value.data,
      label: value.label,
      grad: value.grad,
      graph_nodes: map_size(value.graph),
      summary: "Value(label: #{value.label || "-"}, data: #{number(value.data)}, id: #{value.id})"
    }
  end

  defp value(%Gradients{} = gradients, _depth) do
    values = Gradients.to_map(gradients)

    %{
      kind: "struct",
      domain: "micrograd_gradients",
      module: "MicrogradEx.Gradients",
      output_id: gradients.output_id,
      count: map_size(values),
      values: values,
      summary: "Gradients(output_id: #{gradients.output_id}, count: #{map_size(values)})"
    }
  end

  defp value(%module{} = struct, depth) do
    fields =
      struct
      |> Map.from_struct()
      |> Enum.take(@max_preview)
      |> Enum.map(fn {key, nested} ->
        %{key: Atom.to_string(key), value: nested_value(nested, depth)}
      end)

    %{
      kind: "struct",
      module: inspect(module),
      fields: fields,
      summary: truncate(inspect(struct, limit: @max_preview, printable_limit: @max_string))
    }
  end

  defp value(value, _depth) when is_pid(value) do
    %{kind: "pid", summary: inspect(value)}
  end

  defp value(value, _depth) when is_function(value) do
    %{kind: "function", summary: inspect(value)}
  end

  defp value(value, depth) when is_list(value) do
    %{
      kind: "list",
      length: length(value),
      preview: preview(value, depth),
      summary: truncate(inspect(value, limit: @max_preview, printable_limit: @max_string))
    }
  end

  defp value(value, depth) when is_tuple(value) do
    list = Tuple.to_list(value)

    %{
      kind: "tuple",
      size: tuple_size(value),
      preview: preview(list, depth),
      summary: truncate(inspect(value, limit: @max_preview, printable_limit: @max_string))
    }
  end

  defp value(value, depth) when is_map(value) do
    preview =
      value
      |> Enum.take(@max_preview)
      |> Enum.map(fn {key, nested} ->
        %{key: inspect(key), value: nested_value(nested, depth)}
      end)

    %{
      kind: "map",
      size: map_size(value),
      preview: preview,
      summary: truncate(inspect(value, limit: @max_preview, printable_limit: @max_string))
    }
  end

  defp value(value, _depth) do
    %{
      kind: "term",
      summary: truncate(inspect(value, limit: @max_preview, printable_limit: @max_string))
    }
  end

  defp preview(values, depth) do
    values
    |> Enum.take(@max_preview)
    |> Enum.map(&nested_value(&1, depth))
  end

  defp nested_value(_value, depth) when depth >= 2, do: %{kind: "truncated", summary: "..."}
  defp nested_value(value, depth), do: value(value, depth + 1)

  defp truncate(value) when is_binary(value) do
    if String.length(value) > @max_string do
      String.slice(value, 0, @max_string) <> "..."
    else
      value
    end
  end

  defp number(value) when is_float(value), do: :erlang.float_to_binary(value, decimals: 4)
  defp number(value), do: inspect(value)
end
