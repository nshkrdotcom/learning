defmodule MlVizLab.Instrumentation.RecorderTest do
  use ExUnit.Case, async: true

  alias MlVizLab.Instrumentation.Event
  alias MlVizLab.Instrumentation.Recorder

  test "recorder preserves semantic event order and assigns indexes" do
    recorder =
      start_supervised!({Recorder, run_id: "run", subject_id: "subject", lesson_id: "lesson"})

    assert :ok = Recorder.record(recorder, Event.new(:value_created, %{title: "first"}))
    assert :ok = Recorder.record(recorder, Event.new(:operation_created, %{title: "second"}))

    assert [
             %Event{index: 0, type: :value_created, title: "first"},
             %Event{index: 1, type: :operation_created, title: "second"}
           ] = Recorder.events(recorder)
  end

  test "concurrent recorders do not leak events" do
    one = start_supervised!({Recorder, run_id: "one", subject_id: "subject", lesson_id: "lesson"})
    two = start_supervised!({Recorder, run_id: "two", subject_id: "subject", lesson_id: "lesson"})

    Recorder.record(one, Event.new(:value_created, %{title: "one"}))
    Recorder.record(two, Event.new(:value_created, %{title: "two"}))

    assert [%Event{title: "one"}] = Recorder.events(one)
    assert [%Event{title: "two"}] = Recorder.events(two)
  end

  test "recorder exposes metadata and error finalization" do
    recorder =
      start_supervised!({Recorder, run_id: "run", subject_id: "subject", lesson_id: "lesson"})

    assert Recorder.metadata(recorder).run_id == "run"

    assert %{status: "error", error: %{type: "RuntimeError", message: "bad"}} =
             Recorder.error(recorder, %RuntimeError{message: "bad"})
  end
end
