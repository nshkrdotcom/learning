defmodule MlVizLab.Runs do
  @moduledoc """
  Trace run orchestration.

  Runs are intentionally ephemeral. The current implementation uses LiveView's
  async task supervision and this context centralizes run ids, success payloads,
  and error payloads so stale results can be ignored by the UI.
  """

  alias MlVizLab.Subjects
  alias MlVizLab.Trace.Run

  def new_run_id(subject_id, lesson_id) do
    "#{subject_id}:#{lesson_id}:#{System.unique_integer([:positive, :monotonic])}"
  end

  def generate(subject_id, lesson_id, run_id \\ nil, opts \\ []) do
    run_id = run_id || new_run_id(subject_id, lesson_id)
    started_at = DateTime.utc_now()
    started_ms = System.monotonic_time(:millisecond)

    result =
      try do
        Subjects.run(subject_id, lesson_id, Keyword.put(opts, :run_id, run_id))
      rescue
        exception -> {:error, exception}
      catch
        kind, reason -> {:error, %{type: inspect(kind), message: inspect(reason)}}
      end

    completed_at = DateTime.utc_now()
    duration_ms = System.monotonic_time(:millisecond) - started_ms

    case result do
      {:ok, %Run{} = run} ->
        %{
          run
          | run_id: run_id,
            subject_id: subject_id,
            lesson_id: lesson_id,
            status: "ready",
            started_at: started_at,
            completed_at: completed_at,
            duration_ms: duration_ms
        }

      {:error, error} ->
        error_trace(subject_id, lesson_id, run_id, started_at, completed_at, duration_ms, error)
    end
  end

  defp error_trace(subject_id, lesson_id, run_id, started_at, completed_at, duration_ms, error) do
    Run.new(%{
      run_id: run_id,
      subject_id: subject_id,
      lesson_id: lesson_id,
      title: "Trace failed",
      level: nil,
      description: nil,
      view: "error",
      status: "error",
      sources: [],
      concepts: [],
      checkpoints: [],
      final_graph: %{nodes: [], edges: []},
      events: [],
      stats: %{nodes: 0, edges: 0, steps: 0},
      error: normalize_error(error),
      started_at: started_at,
      completed_at: completed_at,
      duration_ms: duration_ms,
      capabilities: %{}
    })
  end

  defp normalize_error(%{type: type, message: message}),
    do: %{type: to_string(type), message: message}

  defp normalize_error(%{message: message}), do: %{type: "Error", message: message}

  defp normalize_error(%module{} = exception) when is_exception(exception) do
    %{type: module |> Module.split() |> List.last(), message: Exception.message(exception)}
  end

  defp normalize_error(error), do: %{type: "Error", message: inspect(error)}
end
