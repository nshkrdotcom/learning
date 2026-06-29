defmodule MlVizLab.Instrumentation.Recorder do
  @moduledoc """
  Per-run semantic event recorder.

  A recorder is intentionally ephemeral and process-local to a single lesson
  execution. Subject-specific instrumented wrappers append events here while
  still returning the real domain values produced by the underlying library.
  """

  use Agent

  alias MlVizLab.Instrumentation.Event

  def child_spec(opts) do
    %{
      id: {__MODULE__, Keyword.fetch!(opts, :run_id)},
      start: {__MODULE__, :start_link, [opts]},
      restart: :temporary,
      type: :worker
    }
  end

  def start_link(opts) do
    Agent.start_link(fn ->
      %{
        metadata: %{
          run_id: Keyword.fetch!(opts, :run_id),
          subject_id: Keyword.fetch!(opts, :subject_id),
          lesson_id: Keyword.fetch!(opts, :lesson_id)
        },
        events: []
      }
    end)
  end

  def record(recorder, %Event{} = event) do
    Agent.update(recorder, fn state ->
      index = length(state.events)
      event = Event.with_index(event, index)
      %{state | events: state.events ++ [event]}
    end)
  end

  def events(recorder), do: Agent.get(recorder, & &1.events)
  def metadata(recorder), do: Agent.get(recorder, & &1.metadata)
  def snapshot(recorder), do: %{metadata: metadata(recorder), events: events(recorder)}

  def error(_recorder, %module{} = exception) when is_exception(exception) do
    %{
      status: "error",
      error: %{
        type: module |> Module.split() |> List.last(),
        message: Exception.message(exception)
      }
    }
  end

  def error(_recorder, error),
    do: %{status: "error", error: %{type: "Error", message: inspect(error)}}
end
