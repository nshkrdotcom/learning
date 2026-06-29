defmodule MlVizLab.Execution.RuntimeTest do
  use ExUnit.Case, async: true

  alias MlVizLab.Execution.Runtime

  test "runtime genuinely blocks at pause until continue is sent" do
    source = """
    x = 1
    y = x + 2
    y
    """

    session_id = "runtime-test-#{System.unique_integer([:positive])}"

    {:ok, runtime_pid} =
      Runtime.start_link(session_id: session_id, controller_pid: self(), source: source)

    assert_receive {:execution_event,
                    %{type: :runtime_started, session_id: ^session_id, runtime_pid: ^runtime_pid}}

    assert_receive {:execution_event,
                    %{
                      type: :paused,
                      session_id: ^session_id,
                      span: %{line_start: 1},
                      bindings: bindings
                    }}

    refute Map.has_key?(bindings, "x")

    refute_receive {:execution_event, %{type: :binding_snapshot}}, 50

    Runtime.continue(runtime_pid, session_id, "cmd-1")

    assert_receive {:execution_event, %{type: :resumed, command_id: "cmd-1"}}
    assert_receive {:execution_event, %{type: :binding_snapshot, bindings: bindings}}
    assert bindings["x"].value == 1
    refute Map.has_key?(bindings, "y")

    assert_receive {:execution_event,
                    %{type: :paused, span: %{line_start: 2}, bindings: bindings}}

    assert bindings["x"].value == 1
    refute Map.has_key?(bindings, "y")
  end

  test "runtime completion captures return value" do
    source = """
    x = 1
    x + 41
    """

    session_id = "runtime-complete-#{System.unique_integer([:positive])}"

    {:ok, runtime_pid} =
      Runtime.start_link(session_id: session_id, controller_pid: self(), source: source)

    assert_receive {:execution_event, %{type: :runtime_started}}
    assert_receive {:execution_event, %{type: :paused, span: %{line_start: 1}}}
    Runtime.continue(runtime_pid, session_id, "cmd-1")
    assert_receive {:execution_event, %{type: :binding_snapshot}}
    assert_receive {:execution_event, %{type: :paused, span: %{line_start: 2}}}
    Runtime.continue(runtime_pid, session_id, "cmd-2")
    assert_receive {:execution_event, %{type: :binding_snapshot, value: %{value: 42}}}
    assert_receive {:execution_event, %{type: :completed, result: %{value: 42}}}
  end

  test "stop terminates a paused runtime" do
    session_id = "runtime-stop-#{System.unique_integer([:positive])}"

    {:ok, runtime_pid} =
      Runtime.start_link(session_id: session_id, controller_pid: self(), source: "x = 1")

    assert_receive {:execution_event, %{type: :runtime_started}}
    assert_receive {:execution_event, %{type: :paused}}
    Runtime.stop(runtime_pid, session_id, "stop-1")

    assert_receive {:execution_event, %{type: :stopped, command_id: "stop-1"}}
  end

  test "runtime exceptions become structured error events" do
    source = """
    x = 1
    raise "boom"
    """

    session_id = "runtime-error-#{System.unique_integer([:positive])}"

    {:ok, runtime_pid} =
      Runtime.start_link(session_id: session_id, controller_pid: self(), source: source)

    assert_receive {:execution_event, %{type: :runtime_started}}
    assert_receive {:execution_event, %{type: :paused, span: %{line_start: 1}}}
    Runtime.continue(runtime_pid, session_id, "cmd-1")
    assert_receive {:execution_event, %{type: :binding_snapshot}}
    assert_receive {:execution_event, %{type: :paused, span: %{line_start: 2}}}

    Runtime.continue(runtime_pid, session_id, "cmd-2")

    assert_receive {:execution_event,
                    %{
                      type: :error,
                      session_id: ^session_id,
                      runtime_pid: ^runtime_pid,
                      error: %{type: "RuntimeError", message: "boom"}
                    }}
  end

  test "runtime pause timeout becomes a structured error event" do
    previous = Process.flag(:trap_exit, true)
    session_id = "runtime-timeout-#{System.unique_integer([:positive])}"

    try do
      {:ok, runtime_pid} =
        Runtime.start_link(
          session_id: session_id,
          controller_pid: self(),
          source: "x = 1",
          pause_timeout_ms: 1
        )

      assert_receive {:execution_event, %{type: :runtime_started}}
      assert_receive {:execution_event, %{type: :paused}}

      assert_receive {:execution_event,
                      %{
                        type: :error,
                        session_id: ^session_id,
                        runtime_pid: ^runtime_pid,
                        error: %{type: "PauseTimeout", message: "runtime pause timed out"}
                      }}

      assert_receive {:EXIT, ^runtime_pid, :normal}
    after
      Process.flag(:trap_exit, previous)
    end
  end
end
