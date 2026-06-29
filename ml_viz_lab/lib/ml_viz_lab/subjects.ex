defmodule MlVizLab.Subjects do
  @moduledoc """
  Registry and facade for visualized subjects.
  """

  def all do
    Enum.map(enabled_specs(), &metadata/1)
  end

  def default_subject do
    enabled_specs()
    |> Enum.find(&Map.get(&1, :default, false))
    |> case do
      nil ->
        case enabled_specs() do
          [first | _rest] -> Map.fetch!(first, :id)
          [] -> raise "no enabled ML Viz subjects configured"
        end

      spec ->
        Map.fetch!(spec, :id)
    end
  end

  def get(subject_id) when is_binary(subject_id) do
    case Enum.find(enabled_specs(), &(Map.fetch!(&1, :id) == subject_id)) do
      nil -> {:error, unknown_subject(subject_id)}
      spec -> {:ok, spec}
    end
  end

  def get!(subject_id) do
    case get(subject_id) do
      {:ok, spec} -> Map.fetch!(spec, :module)
      {:error, error} -> raise ArgumentError, error.message
    end
  end

  def lessons(subject_id) do
    with {:ok, spec} <- get(subject_id) do
      spec.module.lessons()
    else
      {:error, _error} -> []
    end
  end

  def concepts(subject_id) do
    with {:ok, spec} <- get(subject_id) do
      spec.module.concepts()
    else
      {:error, _error} -> []
    end
  end

  def sources(subject_id, lesson_id) do
    with {:ok, spec} <- get(subject_id) do
      spec.module.sources(lesson_id)
    else
      {:error, _error} -> []
    end
  end

  def capabilities(subject_id) do
    with {:ok, spec} <- get(subject_id) do
      spec.module.capabilities()
    else
      {:error, _error} -> %{}
    end
  end

  def run(subject_id, lesson_id, opts \\ []) do
    with {:ok, spec} <- get(subject_id) do
      spec.module.run(lesson_id, opts)
    end
  end

  def generate_trace(subject_id, lesson_id) do
    case run(subject_id, lesson_id, []) do
      {:ok, trace} -> trace
      {:error, error} -> raise ArgumentError, error_message(error)
    end
  end

  defp enabled_specs do
    :ml_viz_lab
    |> Application.get_env(:subjects, [])
    |> Enum.filter(&Map.get(&1, :enabled, true))
    |> Enum.map(&normalize_spec/1)
  end

  defp normalize_spec(spec) do
    %{
      id: Map.fetch!(spec, :id),
      module: Map.fetch!(spec, :module),
      enabled: Map.get(spec, :enabled, true),
      default: Map.get(spec, :default, false)
    }
  end

  defp metadata(%{module: adapter}) do
    %{
      id: adapter.id(),
      title: adapter.title(),
      description: adapter.description(),
      lessons: adapter.lessons(),
      capabilities: adapter.capabilities()
    }
  end

  defp unknown_subject(subject_id) do
    %{type: "UnknownSubject", message: "unknown subject: #{inspect(subject_id)}"}
  end

  defp error_message(%{message: message}), do: message
  defp error_message(error), do: inspect(error)
end
