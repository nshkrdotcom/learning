defmodule MlVizLab.Execution.SessionTest do
  use ExUnit.Case, async: true

  alias MlVizLab.Execution.Session

  test "session starts, pauses, steps, and completes without leaking state" do
    {:ok, session} =
      start_supervised(
        {Session,
         session_id: "session-a",
         subject_id: "test",
         lesson_id: "simple",
         source: "x = 1\ny = x + 2\ny",
         owner_pid: self()}
      )

    assert_receive {:execution_event, %{type: :session_started, session_id: "session-a"}}
    assert_receive {:execution_event, %{type: :runtime_started, session_id: "session-a"}}

    assert_receive {:execution_event,
                    %{type: :paused, session_id: "session-a", span: %{line_start: 1}}}

    assert %{status: :paused, current_step: 0} = Session.state(session)

    assert :ok = Session.step(session)
    assert_receive {:execution_event, %{type: :binding_snapshot, bindings: bindings}}
    assert bindings["x"].value == 1
    assert_receive {:execution_event, %{type: :paused, span: %{line_start: 2}}}

    assert :ok = Session.step(session)
    assert_receive {:execution_event, %{type: :binding_snapshot, bindings: bindings}}
    assert bindings["y"].value == 3
    assert_receive {:execution_event, %{type: :paused, span: %{line_start: 3}}}

    assert :ok = Session.step(session)
    assert_receive {:execution_event, %{type: :completed, result: %{value: 3}}}
    assert %{status: :completed, current_step: 3} = Session.state(session)
  end

  test "independent sessions do not leak events" do
    {:ok, one} =
      start_supervised(
        {Session,
         session_id: "session-one",
         subject_id: "test",
         lesson_id: "one",
         source: "x = 1",
         owner_pid: self()}
      )

    {:ok, two} =
      start_supervised(
        {Session,
         session_id: "session-two",
         subject_id: "test",
         lesson_id: "two",
         source: "x = 2",
         owner_pid: self()}
      )

    assert_receive {:execution_event, %{type: :session_started, session_id: "session-one"}}
    assert_receive {:execution_event, %{type: :session_started, session_id: "session-two"}}

    wait_for_paused_sessions(MapSet.new(["session-one", "session-two"]))

    Session.step(one)

    assert_receive {:execution_event,
                    %{type: :binding_snapshot, session_id: "session-one", bindings: bindings}}

    assert bindings["x"].value == 1

    assert %{current_bindings: bindings_two} = Session.state(two)
    refute Map.has_key?(bindings_two, "x")
  end

  defp wait_for_paused_sessions(sessions) do
    if MapSet.size(sessions) == 0 do
      :ok
    else
      receive do
        {:execution_event, %{type: :paused, session_id: session_id}} ->
          wait_for_paused_sessions(MapSet.delete(sessions, session_id))

        _other ->
          wait_for_paused_sessions(sessions)
      after
        500 -> flunk("sessions did not pause: #{inspect(MapSet.to_list(sessions))}")
      end
    end
  end
end
