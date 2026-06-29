defmodule MlVizLab.Subjects.Micrograd.Runner do
  @moduledoc """
  Executes Micrograd lessons through instrumented wrappers.
  """

  alias MicrogradEx.NN
  alias MicrogradEx.NN.Neuron
  alias MicrogradEx.Value
  alias MlVizLab.Instrumentation.Recorder
  alias MlVizLab.Subjects.Micrograd.InstrumentedNN
  alias MlVizLab.Subjects.Micrograd.InstrumentedValue, as: IV

  @core_lessons ~w(x_squared repeated_parent single_neuron one_training_step linear_training)

  def core_lesson?(lesson_id), do: lesson_id in @core_lessons

  def run(lesson_id, opts) when lesson_id in @core_lessons do
    run_id = Keyword.get(opts, :run_id, "#{lesson_id}-instrumented")

    {:ok, recorder} =
      Recorder.start_link(run_id: run_id, subject_id: "micrograd", lesson_id: lesson_id)

    try do
      spec = do_run(lesson_id, recorder)
      Map.put(spec, :instrumentation_events, Recorder.events(recorder))
    after
      if Process.alive?(recorder), do: Agent.stop(recorder)
    end
  end

  defp do_run("x_squared", recorder) do
    script = """
    alias MicrogradEx.Value

    x = Value.new(3.0, label: "x")
    y = Value.pow(x, 2, label: "y = x^2")
    gradients = Value.backward(y)
    """

    x = IV.new(recorder, 3.0, script: script, source: "x = Value.new", label: "x")
    y = IV.pow(recorder, x, 2, script: script, source: "y = Value.pow", label: "y = x^2")
    gradients = IV.backward(recorder, y, script: script, source: "gradients = Value.backward")

    graph_spec(
      "x_squared",
      "x squared",
      "Single Values",
      "The smallest useful derivative: y = x^2.",
      script,
      y,
      gradients,
      trace_mode: "instrumented"
    )
  end

  defp do_run("repeated_parent", recorder) do
    script = """
    alias MicrogradEx.Value

    x = Value.new(3.0, label: "x")
    square = Value.mul(x, x, label: "x * x")
    doubled = Value.add(x, x, label: "x + x")
    y = Value.add(square, doubled, label: "x * x + x + x")
    gradients = Value.backward(y)
    """

    x = IV.new(recorder, 3.0, script: script, source: "x = Value.new", label: "x")
    square = IV.mul(recorder, x, x, script: script, source: "square = Value.mul", label: "x * x")

    doubled =
      IV.add(recorder, x, x, script: script, source: "doubled = Value.add", label: "x + x")

    y =
      IV.add(recorder, square, doubled,
        script: script,
        source: "y = Value.add",
        label: "x * x + x + x"
      )

    gradients = IV.backward(recorder, y, script: script, source: "gradients = Value.backward")

    graph_spec(
      "repeated_parent",
      "Repeated parent edge",
      "Operations",
      "Why x * x contributes twice to the same gradient.",
      script,
      y,
      gradients,
      trace_mode: "instrumented"
    )
  end

  defp do_run("single_neuron", recorder) do
    script = """
    alias MicrogradEx.NN
    alias MicrogradEx.NN.Neuron
    alias MicrogradEx.Value

    neuron = Neuron.new(2, weights: [
      Value.new(0.5, label: "w0"),
      Value.new(-1.0, label: "w1")
    ], bias: Value.new(0.25, label: "b"), nonlin: false)
    output = NN.forward(neuron, [2.0, -3.0])
    gradients = Value.backward(output)
    """

    w0 = IV.new(recorder, 0.5, script: script, source: "Value.new(0.5", label: "w0")
    w1 = IV.new(recorder, -1.0, script: script, source: "Value.new(-1.0", label: "w1")
    b = IV.new(recorder, 0.25, script: script, source: "bias: Value.new", label: "b")
    neuron = Neuron.new(2, weights: [w0, w1], bias: b, nonlin: false)

    output =
      InstrumentedNN.forward(recorder, neuron, [2.0, -3.0],
        script: script,
        source: "output = NN.forward"
      )

    gradients =
      IV.backward(recorder, output, script: script, source: "gradients = Value.backward")

    graph_spec(
      "single_neuron",
      "Single neuron",
      "Neural Network",
      "A weighted sum, bias, and parameter gradients.",
      script,
      output,
      gradients,
      view: "network",
      trace_mode: "instrumented"
    )
  end

  defp do_run("one_training_step", recorder) do
    script = """
    alias MicrogradEx.NN
    alias MicrogradEx.NN.Neuron
    alias MicrogradEx.Value

    model = Neuron.new(1, weights: [Value.new(0.0, label: "w")],
      bias: Value.new(0.0, label: "b"), nonlin: false)
    prediction = NN.forward(model, [2.0])
    loss = prediction |> Value.sub(3.0, label: "prediction - target") |> Value.pow(2, label: "loss")
    gradients = Value.backward(loss)
    updated = NN.apply_gradients(model, gradients, 0.1)
    """

    w = IV.new(recorder, 0.0, script: script, source: "Value.new(0.0, label: \"w\")", label: "w")
    b = IV.new(recorder, 0.0, script: script, source: "bias: Value.new", label: "b")
    model = Neuron.new(1, weights: [w], bias: b, nonlin: false)

    prediction =
      InstrumentedNN.forward(recorder, model, [2.0],
        script: script,
        source: "prediction = NN.forward"
      )

    loss =
      prediction
      |> then(
        &IV.sub(recorder, &1, 3.0,
          script: script,
          source: "loss = prediction",
          label: "prediction - target"
        )
      )
      |> then(&IV.pow(recorder, &1, 2, script: script, source: "|> Value.pow", label: "loss"))

    gradients = IV.backward(recorder, loss, script: script, source: "gradients = Value.backward")

    updated =
      InstrumentedNN.apply_gradients(recorder, model, gradients, 0.1,
        script: script,
        source: "updated = NN.apply_gradients"
      )

    graph_spec(
      "one_training_step",
      "One training step",
      "Training",
      "Prediction, loss, backward pass, and immutable parameter update.",
      script,
      loss,
      gradients,
      view: "training",
      params_before: NN.parameters(model),
      params_after: NN.parameters(updated),
      learning_rate: 0.1,
      trace_mode: "instrumented"
    )
  end

  defp do_run("linear_training", _recorder) do
    script = """
    alias MicrogradEx.NN
    alias MicrogradEx.NN.Neuron
    alias MicrogradEx.Value

    model = Neuron.new(1, weights: [Value.new(0.0, label: "w")],
      bias: Value.new(0.0, label: "b"), nonlin: false)

    trained = Enum.reduce(1..10, model, fn _step, model ->
      loss = linear_regression_loss(model)
      gradients = Value.backward(loss)
      NN.apply_gradients(model, gradients, 0.03)
    end)
    """

    initial =
      Neuron.new(1,
        weights: [Value.new(0.0, label: "w")],
        bias: Value.new(0.0, label: "b"),
        nonlin: false
      )

    {_model, epochs} =
      Enum.reduce(1..10, {initial, []}, fn step, {model, epochs} ->
        loss = linear_regression_loss(model)
        gradients = Value.backward(loss)
        updated = NN.apply_gradients(model, gradients, 0.03)
        [weight, bias] = NN.parameters(updated)

        epoch = %{
          step: step,
          loss: loss.data,
          weight: weight.data,
          bias: bias.data,
          gradient_summary: %{
            weight: MicrogradEx.Gradients.get(gradients, hd(NN.parameters(model))),
            bias: MicrogradEx.Gradients.get(gradients, List.last(NN.parameters(model)))
          }
        }

        {updated, epochs ++ [epoch]}
      end)

    %{
      kind: :training_summary,
      id: "linear_training",
      title: "Train y = 2x - 1",
      level: "Training",
      description: "Compressed training steps showing loss falling and parameters moving.",
      script: script,
      view: "training",
      epochs: epochs,
      trace_mode: "instrumented",
      compression: %{
        mode: "epoch_summary",
        detailed: false,
        source_events: "forward_backward_update"
      }
    }
  end

  defp graph_spec(id, title, level, description, script, output, gradients, opts) do
    %{
      kind: :graph,
      id: id,
      title: title,
      level: level,
      description: description,
      script: script,
      output: output,
      gradients: gradients,
      view: Keyword.get(opts, :view, "graph"),
      params_before: Keyword.get(opts, :params_before),
      params_after: Keyword.get(opts, :params_after),
      learning_rate: Keyword.get(opts, :learning_rate),
      trace_mode: Keyword.get(opts, :trace_mode, "instrumented")
    }
  end

  defp linear_regression_loss(model) do
    [
      {-2.0, -5.0},
      {-1.0, -3.0},
      {0.0, -1.0},
      {1.0, 1.0},
      {2.0, 3.0}
    ]
    |> Enum.map(fn {x, target} ->
      model
      |> NN.forward([x])
      |> Value.sub(target)
      |> Value.pow(2)
    end)
    |> Value.sum()
  end
end
