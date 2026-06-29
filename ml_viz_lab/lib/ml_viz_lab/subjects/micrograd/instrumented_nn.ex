defmodule MlVizLab.Subjects.Micrograd.InstrumentedNN do
  @moduledoc """
  Instrumented neural-network helpers that preserve MicrogradEx scalar semantics.
  """

  alias MicrogradEx.NN
  alias MicrogradEx.NN.Layer
  alias MicrogradEx.NN.MLP
  alias MicrogradEx.NN.Neuron
  alias MicrogradEx.Value
  alias MlVizLab.Instrumentation.Event
  alias MlVizLab.Instrumentation.Recorder
  alias MlVizLab.Subjects.Micrograd.InstrumentedValue, as: IV
  alias MlVizLab.Subjects.Micrograd.SourceCatalog

  def forward(recorder, model, inputs, opts \\ []) do
    result = do_forward(recorder, model, inputs, opts)

    Recorder.record(
      recorder,
      Event.new(:nn_forward, %{
        phase: "forward",
        title: "Run neural network forward",
        source:
          SourceCatalog.lesson_source(
            Keyword.fetch!(opts, :script),
            Keyword.get(opts, :source) || "NN.forward"
          ),
        implementation_source:
          SourceCatalog.implementation_source(:nn_forward, Keyword.fetch!(opts, :script)),
        payload: %{output: result},
        concepts: ["parameters", "scalar-values"],
        focus: %{node_id: result.id}
      })
    )

    result
  end

  def apply_gradients(recorder, model, gradients, learning_rate, opts \\ []) do
    before = NN.parameters(model)
    updated = NN.apply_gradients(model, gradients, learning_rate)
    after_params = NN.parameters(updated)

    Recorder.record(
      recorder,
      Event.new(:apply_gradients, %{
        phase: "update",
        title: "Apply gradients",
        source:
          SourceCatalog.lesson_source(
            Keyword.fetch!(opts, :script),
            Keyword.get(opts, :source) || "apply_gradients"
          ),
        implementation_source:
          SourceCatalog.implementation_source(:update, Keyword.fetch!(opts, :script)),
        payload: %{
          before: before,
          after: after_params,
          gradients: gradients,
          learning_rate: learning_rate
        },
        concepts: ["parameters", "sgd", "immutable-gradients"]
      })
    )

    updated
  end

  defp do_forward(recorder, %Neuron{} = neuron, inputs, opts) do
    inputs = coerce_inputs(recorder, inputs, length(neuron.weights), opts)

    activation =
      neuron.weights
      |> Enum.zip(inputs)
      |> Enum.with_index()
      |> Enum.reduce(neuron.bias, fn {{weight, input}, index}, acc ->
        term =
          IV.mul(recorder, weight, input,
            script: Keyword.fetch!(opts, :script),
            source: Keyword.get(opts, :source) || "NN.forward",
            label: "w#{index} * input#{index}"
          )

        IV.add(recorder, acc, term,
          script: Keyword.fetch!(opts, :script),
          source: Keyword.get(opts, :source) || "NN.forward",
          label: "neuron sum #{index}"
        )
      end)

    if neuron.nonlin do
      IV.relu(recorder, activation,
        script: Keyword.fetch!(opts, :script),
        source: Keyword.get(opts, :source) || "NN.forward",
        label: "neuron relu"
      )
    else
      activation
    end
  end

  defp do_forward(recorder, %Layer{} = layer, inputs, opts) do
    layer.neurons
    |> Enum.map(&do_forward(recorder, &1, inputs, opts))
    |> unwrap_single()
  end

  defp do_forward(recorder, %MLP{} = mlp, inputs, opts) do
    mlp.layers
    |> Enum.reduce(inputs, fn layer, layer_inputs ->
      layer.neurons
      |> Enum.map(&do_forward(recorder, &1, layer_inputs, opts))
    end)
    |> unwrap_single()
  end

  defp coerce_inputs(recorder, input, 1, opts) when is_number(input) or is_struct(input, Value) do
    [
      IV.coerce(recorder, input,
        script: Keyword.fetch!(opts, :script),
        source: Keyword.get(opts, :source),
        label: "input0"
      )
    ]
  end

  defp coerce_inputs(recorder, inputs, expected_count, opts) when is_list(inputs) do
    if length(inputs) != expected_count do
      raise ArgumentError, "expected #{expected_count} inputs, got #{length(inputs)}"
    end

    inputs
    |> Enum.with_index()
    |> Enum.map(fn {input, index} ->
      IV.coerce(recorder, input,
        script: Keyword.fetch!(opts, :script),
        source: Keyword.get(opts, :source),
        label: "input#{index}"
      )
    end)
  end

  defp unwrap_single([only]), do: only
  defp unwrap_single(values), do: values
end
