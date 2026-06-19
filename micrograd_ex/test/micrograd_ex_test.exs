defmodule MicrogradExTest do
  use ExUnit.Case
  doctest MicrogradEx

  test "greets the world" do
    assert MicrogradEx.hello() == :world
  end
end
