defmodule MlVizLab.Subjects.Micrograd.InstrumentedValue do
  @moduledoc """
  MicrogradEx.Value wrapper that records semantic events as real operations run.
  """

  alias MicrogradEx.Gradients
  alias MicrogradEx.Value
  alias MlVizLab.Instrumentation.Event
  alias MlVizLab.Instrumentation.Recorder
  alias MlVizLab.Subjects.Micrograd.SourceCatalog

  def new(recorder, data, opts \\ []) when is_number(data) do
    value = Value.new(data, label: Keyword.get(opts, :label))

    record_value(recorder, :value_created, "initialization", value, [], :leaf, opts)
    value
  end

  def coerce(recorder, value, opts \\ [])

  def coerce(_recorder, %Value{} = value, _opts), do: value

  def coerce(recorder, number, opts) when is_number(number) do
    label = Keyword.get(opts, :label) || number_label(number)
    new(recorder, number, Keyword.put(opts, :label, label))
  end

  def add(recorder, left, right, opts \\ []) do
    binary_op(recorder, :+, &Value.add/3, left, right, opts)
  end

  def sub(recorder, left, right, opts \\ []) do
    binary_op(recorder, :-, &Value.sub/3, left, right, opts)
  end

  def mul(recorder, left, right, opts \\ []) do
    binary_op(recorder, :*, &Value.mul/3, left, right, opts)
  end

  def neg(recorder, value, opts \\ []) do
    value = coerce(recorder, value, constant_opts(opts, "value"))
    result = Value.neg(value, label_opts(opts))

    record_value(recorder, :operation_created, "forward", result, [value], :neg, opts)
    result
  end

  def pow(recorder, value, exponent, opts \\ []) when is_number(exponent) do
    value = coerce(recorder, value, constant_opts(opts, "base"))
    result = Value.pow(value, exponent, label_opts(opts))

    record_value(recorder, :operation_created, "forward", result, [value], {:pow, exponent}, opts)
    result
  end

  def divide(recorder, left, right, opts \\ []) do
    reciprocal =
      pow(recorder, right, -1.0,
        script: Keyword.fetch!(opts, :script),
        source: Keyword.get(opts, :source),
        label: Keyword.get(opts, :reciprocal_label, "reciprocal")
      )

    mul(recorder, left, reciprocal, opts)
  end

  def relu(recorder, value, opts \\ []) do
    value = coerce(recorder, value, constant_opts(opts, "value"))
    result = Value.relu(value, label_opts(opts))

    record_value(recorder, :operation_created, "forward", result, [value], :relu, opts)
    result
  end

  def sum(recorder, values, initial \\ nil, opts \\ []) when is_list(values) do
    initial =
      if initial do
        coerce(recorder, initial, constant_opts(opts, "sum start"))
      else
        new(recorder, 0.0,
          script: Keyword.fetch!(opts, :script),
          source: Keyword.get(opts, :source),
          label: "sum start"
        )
      end

    Enum.reduce(values, initial, fn value, acc ->
      add(recorder, acc, value,
        script: Keyword.fetch!(opts, :script),
        source: Keyword.get(opts, :source),
        label: Keyword.get(opts, :label)
      )
    end)
  end

  def backward(recorder, %Value{} = output, opts \\ []) do
    gradients = Value.backward(output)

    Recorder.record(
      recorder,
      Event.new(:backward_called, %{
        phase: "backward",
        title: "Call backward",
        source: lesson_source(opts, "backward"),
        implementation_source: implementation_source(opts, :backward),
        payload: %{output: output, gradients: gradients},
        concepts: ["chain-rule", "immutable-gradients"]
      })
    )

    gradients
  end

  defp binary_op(recorder, op, fun, left, right, opts) do
    left = coerce(recorder, left, constant_opts(opts, "left"))
    right = coerce(recorder, right, constant_opts(opts, "right"))
    result = fun.(left, right, label_opts(opts))

    record_value(recorder, :operation_created, "forward", result, [left, right], op, opts)
    result
  end

  defp record_value(recorder, type, phase, %Value{} = value, inputs, op, opts) do
    node = Map.fetch!(value.graph, value.id)

    Recorder.record(
      recorder,
      Event.new(type, %{
        phase: phase,
        title: Keyword.get(opts, :title) || title_for(type, node),
        source: lesson_source(opts, node.label || op_token(op)),
        implementation_source: implementation_source(opts, op),
        payload: %{
          output: value,
          output_node: node,
          input_ids: Enum.map(inputs, & &1.id),
          edge_ids: edge_ids(node),
          op: op
        },
        concepts: concepts_for(op),
        focus: %{node_id: value.id}
      })
    )
  end

  defp edge_ids(node) do
    node.parents
    |> Enum.with_index()
    |> Enum.map(fn {edge, index} -> "#{edge.parent_id}->#{node.id}:#{index}" end)
  end

  defp label_opts(opts), do: opts |> Keyword.take([:label])

  defp constant_opts(opts, side) do
    [
      script: Keyword.fetch!(opts, :script),
      source: Keyword.get(opts, :source),
      label: Keyword.get(opts, :"#{side}_label")
    ]
  end

  defp lesson_source(opts, token) do
    SourceCatalog.lesson_source(
      Keyword.fetch!(opts, :script),
      Keyword.get(opts, :source) || token
    )
  end

  defp implementation_source(opts, op) do
    SourceCatalog.implementation_source(op, Keyword.fetch!(opts, :script))
  end

  defp title_for(:value_created, node), do: "Create #{node.label || "leaf"}"
  defp title_for(:operation_created, node), do: "Create #{op_token(node.op)} node"

  defp concepts_for(:leaf), do: ["scalar-values"]
  defp concepts_for(:relu), do: ["local-derivatives", "relu-gate"]
  defp concepts_for(:*), do: ["local-derivatives"]
  defp concepts_for({:pow, _}), do: ["local-derivatives", "chain-rule"]
  defp concepts_for(_op), do: ["local-derivatives", "scalar-values"]

  defp op_token(:leaf), do: "Value.new"
  defp op_token(:+), do: "Value.add"
  defp op_token(:-), do: "Value.sub"
  defp op_token(:*), do: "Value.mul"
  defp op_token(:neg), do: "Value.neg"
  defp op_token(:relu), do: "Value.relu"
  defp op_token({:pow, _}), do: "Value.pow"
  defp op_token(op), do: inspect(op)

  defp number_label(value) when is_integer(value), do: Integer.to_string(value)
  defp number_label(value) when is_float(value), do: :erlang.float_to_binary(value, decimals: 4)

  def gradients_to_map(%Gradients{} = gradients), do: Gradients.to_map(gradients)
end
