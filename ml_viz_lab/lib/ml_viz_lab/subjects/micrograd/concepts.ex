defmodule MlVizLab.Subjects.Micrograd.Concepts do
  @moduledoc false

  def all do
    [
      concept(
        "scalar-values",
        "Scalar values",
        "Every visible block is one scalar number. Neural networks here are built by composing many small scalar operations."
      ),
      concept(
        "local-derivatives",
        "Local derivatives",
        "Each edge stores how sensitive one operation output is to one direct parent at the moment the forward value is created."
      ),
      concept(
        "chain-rule",
        "Chain rule",
        "Backward propagation multiplies the upstream gradient by each edge's local derivative, then adds that contribution to the parent."
      ),
      concept(
        "topological-order",
        "Topological order",
        "Backprop needs an order where a node receives all downstream contributions before it sends gradients to its parents."
      ),
      concept(
        "gradient-accumulation",
        "Gradient accumulation",
        "When the same value is used more than once, each path contributes a separate amount to the same gradient total."
      ),
      concept(
        "relu-gate",
        "ReLU gate",
        "ReLU passes positive values with local derivative 1 and stops non-positive values with local derivative 0."
      ),
      concept(
        "immutable-gradients",
        "Immutable gradients",
        "MicrogradEx returns a Gradients table instead of mutating Value structs. The table is the source of truth."
      ),
      concept(
        "parameters",
        "Parameters",
        "Weights and biases are ordinary scalar Values. Training changes them by creating updated Values."
      ),
      concept(
        "loss",
        "Loss",
        "The loss is a scalar summary of prediction error. Backward starts from this scalar output."
      ),
      concept(
        "sgd",
        "Gradient descent",
        "A parameter update subtracts learning_rate * gradient from each parameter value."
      )
    ]
  end

  defp concept(id, title, body), do: %{id: id, title: title, body: body}
end
