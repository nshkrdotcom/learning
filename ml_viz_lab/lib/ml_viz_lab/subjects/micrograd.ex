defmodule MlVizLab.Subjects.Micrograd do
  @moduledoc false

  @behaviour MlVizLab.Subjects.Adapter

  alias MicrogradEx.NN
  alias MicrogradEx.NN.Layer
  alias MicrogradEx.NN.MLP
  alias MicrogradEx.NN.Neuron
  alias MicrogradEx.Value
  alias MlVizLab.Subjects.Micrograd.Concepts
  alias MlVizLab.Subjects.Micrograd.DomainSnapshot
  alias MlVizLab.Subjects.Micrograd.LiveLesson
  alias MlVizLab.Subjects.Micrograd.Runner
  alias MlVizLab.Subjects.Micrograd.SourceCatalog
  alias MlVizLab.Subjects.Micrograd.TraceBuilder

  @impl true
  def id, do: "micrograd"

  @impl true
  def title, do: "MicrogradEx"

  @impl true
  def description do
    "Scalar automatic differentiation, neural networks, and immutable gradient updates on the BEAM."
  end

  @impl true
  def capabilities do
    %{
      trace_modes: ["instrumented"],
      source_modes: ["lesson", "implementation"],
      views: ["graph", "network", "training"],
      compressed_training: true
    }
  end

  @impl true
  def lessons do
    [
      lesson(
        "x_squared",
        "x squared",
        "Single Values",
        "The smallest useful derivative: y = x^2."
      ),
      lesson(
        "sum_rule",
        "Sum rule",
        "Single Values",
        "Two independent inputs flow into one addition, and each gets gradient 1."
      ),
      lesson(
        "chain_rule_composed",
        "Composed chain rule",
        "Single Values",
        "A nested multiply and power expression shows local derivatives multiplying through a path."
      ),
      lesson(
        "karpathy_sanity",
        "Karpathy sanity expression",
        "Operations",
        "A mixed expression with addition, multiplication, ReLU, and shared dependencies."
      ),
      lesson(
        "repeated_parent",
        "Repeated parent edge",
        "Operations",
        "Why x * x contributes twice to the same gradient."
      ),
      lesson(
        "relu_gate",
        "ReLU gate",
        "Operations",
        "Positive, zero, and negative ReLU behavior in one graph."
      ),
      lesson(
        "single_neuron",
        "Single neuron",
        "Neural Network",
        "A weighted sum, bias, and parameter gradients."
      ),
      lesson(
        "one_layer_forward",
        "One layer forward",
        "Neural Network",
        "Three neurons process the same input and expose a wider scalar graph."
      ),
      lesson(
        "tiny_mlp",
        "Tiny MLP",
        "Neural Network",
        "A small multilayer perceptron decomposed into scalar operations."
      ),
      lesson(
        "mlp_loss_backward",
        "MLP loss and backward",
        "Neural Network",
        "A tiny MLP prediction becomes a squared loss with gradients through every parameter."
      ),
      lesson(
        "one_training_step",
        "One training step",
        "Training",
        "Prediction, loss, backward pass, and immutable parameter update."
      ),
      lesson(
        "linear_training",
        "Train y = 2x - 1",
        "Training",
        "Compressed training steps showing loss falling and parameters moving."
      )
    ]
  end

  @impl true
  def concepts, do: Concepts.all()

  @impl true
  def sources(lesson_id) do
    lesson_id
    |> run_lesson([])
    |> Map.fetch!(:script)
    |> SourceCatalog.sources()
  rescue
    _exception -> []
  end

  @impl true
  def run(lesson_id, opts) do
    trace =
      lesson_id
      |> run_lesson(opts)
      |> TraceBuilder.build()
      |> Map.put(:capabilities, capabilities())
      |> SourceCatalog.verify_trace!()

    {:ok, %{trace | run_id: Keyword.get(opts, :run_id, trace.run_id)}}
  rescue
    exception ->
      {:error, exception}
  end

  def generate_trace(lesson_id) do
    case run(lesson_id, []) do
      {:ok, trace} -> trace
      {:error, error} -> raise error
    end
  end

  def live_source(lesson_id), do: LiveLesson.source!(lesson_id)
  def live_domain_adapter, do: DomainSnapshot

  defp lesson(id, title, level, description) do
    %{id: id, title: title, level: level, description: description}
  end

  defp run_lesson(lesson_id, opts) do
    if Runner.core_lesson?(lesson_id) do
      Runner.run(lesson_id, opts)
    else
      lesson_id
      |> run_lesson()
      |> Map.put(:trace_mode, "compatibility")
    end
  end

  defp run_lesson("x_squared") do
    script = """
    alias MicrogradEx.Value

    x = Value.new(3.0, label: "x")
    y = Value.pow(x, 2, label: "y = x^2")
    gradients = Value.backward(y)
    """

    x = Value.new(3.0, label: "x")
    y = Value.pow(x, 2, label: "y = x^2")
    gradients = Value.backward(y)

    graph_spec(
      "x_squared",
      "x squared",
      "Single Values",
      "The smallest useful derivative: y = x^2.",
      script,
      y,
      gradients
    )
  end

  defp run_lesson("sum_rule") do
    script = """
    alias MicrogradEx.Value

    a = Value.new(2.0, label: "a")
    b = Value.new(-5.0, label: "b")
    y = Value.add(a, b, label: "a + b")
    gradients = Value.backward(y)
    """

    a = Value.new(2.0, label: "a")
    b = Value.new(-5.0, label: "b")
    y = Value.add(a, b, label: "a + b")
    gradients = Value.backward(y)

    graph_spec(
      "sum_rule",
      "Sum rule",
      "Single Values",
      "Two independent inputs flow into one addition, and each gets gradient 1.",
      script,
      y,
      gradients
    )
  end

  defp run_lesson("chain_rule_composed") do
    script = """
    alias MicrogradEx.Value

    x = Value.new(2.0, label: "x")
    doubled = Value.mul(x, 2.0, label: "2x")
    y = Value.pow(doubled, 3, label: "(2x)^3")
    gradients = Value.backward(y)
    """

    x = Value.new(2.0, label: "x")
    doubled = Value.mul(x, 2.0, label: "2x")
    y = Value.pow(doubled, 3, label: "(2x)^3")
    gradients = Value.backward(y)

    graph_spec(
      "chain_rule_composed",
      "Composed chain rule",
      "Single Values",
      "A nested multiply and power expression shows local derivatives multiplying through a path.",
      script,
      y,
      gradients
    )
  end

  defp run_lesson("karpathy_sanity") do
    script = """
    alias MicrogradEx.Value

    x = Value.new(-4.0, label: "x")
    z = Value.add(Value.add(Value.mul(2.0, x, label: "2x"), 2.0, label: "2x + 2"), x, label: "z")
    q = Value.add(Value.relu(z, label: "relu(z)"), Value.mul(z, x, label: "z * x"), label: "q")
    h = Value.relu(Value.mul(z, z, label: "z * z"), label: "h")
    y = Value.add(Value.add(h, q, label: "h + q"), Value.mul(q, x, label: "q * x"), label: "y")
    gradients = Value.backward(y)
    """

    x = Value.new(-4.0, label: "x")
    z = Value.add(Value.add(Value.mul(2.0, x, label: "2x"), 2.0, label: "2x + 2"), x, label: "z")
    q = Value.add(Value.relu(z, label: "relu(z)"), Value.mul(z, x, label: "z * x"), label: "q")
    h = Value.relu(Value.mul(z, z, label: "z * z"), label: "h")
    y = Value.add(Value.add(h, q, label: "h + q"), Value.mul(q, x, label: "q * x"), label: "y")
    gradients = Value.backward(y)

    graph_spec(
      "karpathy_sanity",
      "Karpathy sanity expression",
      "Operations",
      "A mixed expression with addition, multiplication, ReLU, and shared dependencies.",
      script,
      y,
      gradients
    )
  end

  defp run_lesson("repeated_parent") do
    script = """
    alias MicrogradEx.Value

    x = Value.new(3.0, label: "x")
    square = Value.mul(x, x, label: "x * x")
    doubled = Value.add(x, x, label: "x + x")
    y = Value.add(square, doubled, label: "x * x + x + x")
    gradients = Value.backward(y)
    """

    x = Value.new(3.0, label: "x")
    square = Value.mul(x, x, label: "x * x")
    doubled = Value.add(x, x, label: "x + x")
    y = Value.add(square, doubled, label: "x * x + x + x")
    gradients = Value.backward(y)

    graph_spec(
      "repeated_parent",
      "Repeated parent edge",
      "Operations",
      "Why x * x contributes twice to the same gradient.",
      script,
      y,
      gradients
    )
  end

  defp run_lesson("relu_gate") do
    script = """
    alias MicrogradEx.Value

    negative = Value.new(-2.0, label: "negative")
    zero = Value.new(0.0, label: "zero")
    positive = Value.new(2.0, label: "positive")
    output = Value.sum([
      Value.relu(negative, label: "relu negative"),
      Value.relu(zero, label: "relu zero"),
      Value.relu(positive, label: "relu positive")
    ], Value.new(0.0, label: "sum start"))
    gradients = Value.backward(output)
    """

    negative = Value.new(-2.0, label: "negative")
    zero = Value.new(0.0, label: "zero")
    positive = Value.new(2.0, label: "positive")

    output =
      Value.sum(
        [
          Value.relu(negative, label: "relu negative"),
          Value.relu(zero, label: "relu zero"),
          Value.relu(positive, label: "relu positive")
        ],
        Value.new(0.0, label: "sum start")
      )

    gradients = Value.backward(output)

    graph_spec(
      "relu_gate",
      "ReLU gate",
      "Operations",
      "Positive, zero, and negative ReLU behavior in one graph.",
      script,
      output,
      gradients
    )
  end

  defp run_lesson("single_neuron") do
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

    neuron =
      Neuron.new(2,
        weights: [
          Value.new(0.5, label: "w0"),
          Value.new(-1.0, label: "w1")
        ],
        bias: Value.new(0.25, label: "b"),
        nonlin: false
      )

    output = NN.forward(neuron, [2.0, -3.0])
    gradients = Value.backward(output)

    graph_spec(
      "single_neuron",
      "Single neuron",
      "Neural Network",
      "A weighted sum, bias, and parameter gradients.",
      script,
      output,
      gradients,
      view: "network"
    )
  end

  defp run_lesson("tiny_mlp") do
    script = """
    alias MicrogradEx.NN
    alias MicrogradEx.NN.MLP
    alias MicrogradEx.Value

    mlp = MLP.new(2, [2, 1], seed: {7, 8, 9})
    prediction = NN.forward(mlp, [1.0, -2.0])
    gradients = Value.backward(prediction)
    """

    mlp = MLP.new(2, [2, 1], seed: {7, 8, 9})
    prediction = NN.forward(mlp, [1.0, -2.0])
    gradients = Value.backward(prediction)

    graph_spec(
      "tiny_mlp",
      "Tiny MLP",
      "Neural Network",
      "A small multilayer perceptron decomposed into scalar operations.",
      script,
      prediction,
      gradients,
      view: "network"
    )
  end

  defp run_lesson("one_layer_forward") do
    script = """
    alias MicrogradEx.NN
    alias MicrogradEx.NN.Layer
    alias MicrogradEx.Value

    layer = Layer.new(2, 3, seed: {21, 22, 23}, nonlin: true)
    outputs = Layer.forward_many(layer, [1.5, -0.5])
    combined = Value.sum(outputs, Value.new(0.0, label: "layer sum start"))
    gradients = Value.backward(combined)
    """

    layer = Layer.new(2, 3, seed: {21, 22, 23}, nonlin: true)
    outputs = Layer.forward_many(layer, [1.5, -0.5])
    combined = Value.sum(outputs, Value.new(0.0, label: "layer sum start"))
    gradients = Value.backward(combined)

    graph_spec(
      "one_layer_forward",
      "One layer forward",
      "Neural Network",
      "Three neurons process the same input and expose a wider scalar graph.",
      script,
      combined,
      gradients,
      view: "network"
    )
  end

  defp run_lesson("mlp_loss_backward") do
    script = """
    alias MicrogradEx.NN
    alias MicrogradEx.NN.MLP
    alias MicrogradEx.Value

    mlp = MLP.new(2, [2, 1], seed: {31, 32, 33})
    prediction = NN.forward(mlp, [0.75, -1.25])
    loss = prediction |> Value.sub(1.0, label: "prediction - target") |> Value.pow(2, label: "mlp loss")
    gradients = Value.backward(loss)
    """

    mlp = MLP.new(2, [2, 1], seed: {31, 32, 33})
    prediction = NN.forward(mlp, [0.75, -1.25])

    loss =
      prediction
      |> Value.sub(1.0, label: "prediction - target")
      |> Value.pow(2, label: "mlp loss")

    gradients = Value.backward(loss)

    graph_spec(
      "mlp_loss_backward",
      "MLP loss and backward",
      "Neural Network",
      "A tiny MLP prediction becomes a squared loss with gradients through every parameter.",
      script,
      loss,
      gradients,
      view: "network"
    )
  end

  defp run_lesson("one_training_step") do
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

    model =
      Neuron.new(1,
        weights: [Value.new(0.0, label: "w")],
        bias: Value.new(0.0, label: "b"),
        nonlin: false
      )

    prediction = NN.forward(model, [2.0])

    loss =
      prediction |> Value.sub(3.0, label: "prediction - target") |> Value.pow(2, label: "loss")

    gradients = Value.backward(loss)
    updated = NN.apply_gradients(model, gradients, 0.1)

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
      learning_rate: 0.1
    )
  end

  defp run_lesson("linear_training") do
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
          bias: bias.data
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
      epochs: epochs
    }
  end

  defp run_lesson(other) do
    raise ArgumentError, "unknown Micrograd lesson: #{inspect(other)}"
  end

  defp graph_spec(id, title, level, description, script, output, gradients, opts \\ []) do
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
      learning_rate: Keyword.get(opts, :learning_rate)
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
