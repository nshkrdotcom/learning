defmodule MlVizLab.Subjects.Micrograd.LiveLesson do
  @moduledoc """
  Live-executable Micrograd lesson sources.
  """

  def source!("x_squared") do
    """
    x = MicrogradEx.Value.new(3.0, label: "x")
    y = MicrogradEx.Value.pow(x, 2, label: "y = x^2")
    gradients = MicrogradEx.Value.backward(y)
    """
  end

  def source!(lesson_id),
    do: raise(ArgumentError, "no live Micrograd lesson: #{inspect(lesson_id)}")
end
