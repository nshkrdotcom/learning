# MicrogradEx

MicrogradEx is an Elixir-native port of Andrej Karpathy's
[micrograd](https://github.com/karpathy/micrograd): a tiny scalar automatic
differentiation engine with a small neural-network library built on top.

The goal is educational clarity. The engine works over scalar values, builds a
dynamic computation graph during the forward pass, and runs reverse-mode
automatic differentiation during the backward pass. The neural-network layer
then composes those scalar operations into neurons, layers, and multi-layer
perceptrons.

## What Is Included

This project ports the full core scope of original micrograd:

* `MicrogradEx.Value` for scalar values and operations:
  `add`, `sub`, `mul`, `divide`, `pow`, `relu`, `neg`, and `sum`.
* `MicrogradEx.Gradients` for immutable gradient results from `backward/1`.
* `MicrogradEx.NN.Neuron`, `MicrogradEx.NN.Layer`, and `MicrogradEx.NN.MLP`.
* `MicrogradEx.NN.apply_gradients/3` for SGD-style immutable parameter updates.
* Tests that cover the original micrograd reference expressions, gradient
  accumulation through shared graph nodes, ReLU behavior, MLP structure, seeded
  initialization, immutable parameter updates, and a real training loop.

It intentionally does not try to become a tensor library. Like the original,
every neuron is decomposed into scalar additions, multiplications, powers, and
activations so the chain rule remains visible.

## Why The API Is Different From Python

Python micrograd relies on mutable objects:

```python
loss.backward()
print(weight.grad)
weight.data += -learning_rate * weight.grad
```

Elixir data is immutable, so MicrogradEx uses returned values instead:

```elixir
loss = ...
gradients = MicrogradEx.Value.backward(loss)
weight_gradient = MicrogradEx.Value.grad(weight, gradients)
model = MicrogradEx.NN.apply_gradients(model, gradients, learning_rate)
```

That is the central design difference. The math is the same, but the state flow
is explicit and functional:

1. The forward pass returns a new `Value` containing the graph needed for
   backpropagation.
2. The backward pass returns a new `Gradients` table.
3. A training step returns a new model with updated parameter values.

This is more idiomatic Elixir than trying to simulate Python's in-place
mutation.

## Quick Start

Run the test suite:

```bash
mix test
```

Open an interactive shell:

```bash
iex -S mix
```

Compute a derivative:

```elixir
alias MicrogradEx.Value

x = Value.new(3.0, label: "x")
y = Value.pow(x, 2)

gradients = Value.backward(y)

y.data
#=> 9.0

Value.grad(x, gradients)
#=> 6.0
```

The gradient is `6.0` because `d(x^2)/dx = 2x`, and `x = 3`.

## Livebook Demo

The main demo is `notebooks/micrograd_demo.livemd`.

It recreates the official micrograd learning workflow in pure Elixir:

* generate a deterministic two-moons dataset;
* train `MLP.new(2, [16, 16, 1])`;
* confirm the official `337` parameter count;
* optimize max-margin loss with L2 regularization;
* plot loss and accuracy;
* visualize the learned decision boundary.

Open the notebook in Livebook and run it top-to-bottom. The notebook installs
Kino and Vega-Lite locally with `Mix.install/2`; those visualization packages
are not runtime dependencies of the core library.

## Port Of The Original Scalar Example

The original README contains a deliberately mixed expression that uses
addition, multiplication, power, division, negation, and ReLU. In Elixir the
same computation is written with functions because Elixir does not support
operator overloading for custom structs:

```elixir
alias MicrogradEx.Value

a = Value.new(-4.0)
b = Value.new(2.0)

c = Value.add(a, b)
d = Value.add(Value.mul(a, b), Value.pow(b, 3))

c = Value.add(Value.add(c, c), 1.0)
c = Value.add(Value.add(Value.add(c, 1.0), c), Value.neg(a))

d = Value.add(Value.add(d, Value.mul(d, 2.0)), Value.relu(Value.add(b, a)))
d = Value.add(Value.add(d, Value.mul(3.0, d)), Value.relu(Value.sub(b, a)))

e = Value.sub(c, d)
f = Value.pow(e, 2)
g = Value.add(Value.divide(f, 2.0), Value.divide(10.0, f))

gradients = Value.backward(g)

Float.round(g.data, 4)
#=> 24.7041

Float.round(Value.grad(a, gradients), 4)
#=> 138.8338

Float.round(Value.grad(b, gradients), 4)
#=> 645.5773
```

## How Backpropagation Works Here

Each `Value` stores:

* `data`: the scalar result of the forward computation.
* `graph`: a map of graph node ids to immutable node records.
* `id`: the id of the current output node.
* `label`: optional debugging text.
* `grad`: a display-only field, left at `0.0` unless you explicitly annotate a
  value with `Value.with_grad/2`.

Each operation records local derivative edges. For example:

* `a + b` records `da = 1` and `db = 1`.
* `a * b` records `da = b` and `db = a`.
* `x ** n` records `dx = n * x ** (n - 1)`.
* `relu(x)` records `dx = 1` when the output is positive and `0` otherwise.

`Value.backward(output)` topologically walks the graph from output to leaves.
At each node it multiplies the upstream gradient by the local derivative for
each parent edge and adds that contribution into the gradient table. Repeated
parent edges are preserved, so expressions like `x * x` correctly produce
`2x`, and `x + x` correctly produces `2`.

## Neural Networks

The neural-network API mirrors the original project:

```elixir
alias MicrogradEx.NN
alias MicrogradEx.NN.MLP

model = MLP.new(3, [4, 4, 1], seed: {101, 102, 103})

prediction = NN.forward(model, [2.0, -1.0, 0.5])

prediction.data
#=> some scalar output

NN.parameter_count(model)
#=> 41
```

For an MLP with input width `3` and layer widths `[4, 4, 1]`, the parameter
count is:

* first layer: `4 * 3` weights plus `4` biases = `16`;
* second layer: `4 * 4` weights plus `4` biases = `20`;
* final layer: `1 * 4` weights plus `1` bias = `5`;
* total = `41`.

Hidden layers use ReLU. The final layer is linear, matching original micrograd.

## A Tiny Training Loop

This example trains a single linear neuron to learn `y = 2x - 1`.

```elixir
alias MicrogradEx.NN
alias MicrogradEx.NN.Neuron
alias MicrogradEx.Value

data = [
  {-2.0, -5.0},
  {-1.0, -3.0},
  {0.0, -1.0},
  {1.0, 1.0},
  {2.0, 3.0}
]

model =
  Neuron.new(1,
    weights: [0.0],
    bias: 0.0,
    nonlin: false
  )

loss = fn model ->
  data
  |> Enum.map(fn {x, target} ->
    model
    |> NN.forward([x])
    |> Value.sub(target)
    |> Value.pow(2)
  end)
  |> Value.sum()
end

trained =
  Enum.reduce(1..80, model, fn _step, model ->
    current_loss = loss.(model)
    gradients = Value.backward(current_loss)
    NN.apply_gradients(model, gradients, 0.03)
  end)

[weight, bias] = NN.parameters(trained)

weight.data
#=> approximately 2.0

bias.data
#=> approximately -1.0
```

Notice that no `zero_grad` call is needed. Each backward pass returns a fresh
gradient table, so gradients cannot accidentally accumulate across training
steps unless you explicitly combine them yourself.

## Project Layout

```text
lib/
  micrograd_ex.ex          # small public convenience facade
  micrograd_ex/value.ex    # scalar graph values and arithmetic operations
  micrograd_ex/gradients.ex # reverse-mode autodiff result and graph traversal
  micrograd_ex/nn.ex       # Neuron, Layer, MLP, and immutable SGD updates

test/
  value_test.exs           # scalar autodiff and original reference expressions
  nn_test.exs              # neural-network behavior and training
```

## Development

Format code:

```bash
mix format
```

Run tests:

```bash
mix test
```

The tests are intentionally behavioral rather than shallow smoke tests. They
verify the numeric forward values and gradients from the original micrograd
examples, graph edge accumulation, and a working gradient-descent training loop.

## Attribution

This project is a port of the core ideas and API shape from
[micrograd](https://github.com/karpathy/micrograd), created by Andrej Karpathy.
The original project is MIT licensed and is designed for educational use.

MicrogradEx adapts the implementation to Elixir's functional programming model:
immutable values, explicit gradient tables, and model updates that return new
structs instead of mutating parameters in place.

## License

The original micrograd license notice is preserved in this repository's
`LICENSE` file. This Elixir port is provided under the same MIT License terms.
