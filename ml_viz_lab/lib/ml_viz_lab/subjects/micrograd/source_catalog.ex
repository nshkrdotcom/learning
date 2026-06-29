defmodule MlVizLab.Subjects.Micrograd.SourceCatalog do
  @moduledoc false

  @source_root Path.expand("../micrograd_ex/lib", File.cwd!())

  @library_files [
    {"value.ex", "value.ex", "micrograd_ex/value.ex"},
    {"gradients.ex", "gradients.ex", "micrograd_ex/gradients.ex"},
    {"nn.ex", "nn.ex", "micrograd_ex/nn.ex"},
    {"micrograd_ex.ex", "micrograd_ex.ex", "micrograd_ex.ex"}
  ]

  def sources(lesson_source) do
    [
      %{
        id: "lesson.ex",
        title: "lesson.ex",
        path: "generated lesson script",
        source: lesson_source
      }
      | Enum.map(@library_files, fn {id, title, relative} ->
          path = Path.join(@source_root, relative)

          %{
            id: id,
            title: title,
            path: Path.join(["..", "micrograd_ex", "lib", relative]),
            source: File.read!(path)
          }
        end)
    ]
  end

  def lesson_line(script, token) do
    line_containing(script, token)
  end

  def implementation_source(:leaf, _script), do: source_ref("value.ex", "def new")

  def implementation_source(:topology, _script),
    do: source_ref("gradients.ex", "def topological_ids")

  def implementation_source(:backward, _script), do: source_ref("gradients.ex", "def backward")
  def implementation_source(:update, _script), do: source_ref("nn.ex", "def apply_gradients")
  def implementation_source(:nn_forward, _script), do: source_ref("nn.ex", "def forward")
  def implementation_source(:sum, _script), do: source_ref("value.ex", "def sum")
  def implementation_source(:divide, _script), do: source_ref("value.ex", "def divide")

  def implementation_source(op, _script) when op in [:+, :-, :*, :neg, :relu] do
    source_ref("value.ex", "def #{op_name(op)}")
  end

  def implementation_source({:pow, _}, _script), do: source_ref("value.ex", "def pow")

  def implementation_source(_op, _script), do: source_ref("lesson.ex", "", 1)

  def lesson_source(script, token) do
    source_ref("lesson.ex", token, lesson_line(script, token))
  end

  def source_ref(file_id, token, fallback_line \\ nil) do
    line =
      fallback_line ||
        case file_id do
          "lesson.ex" -> 1
          id -> library_source(id) |> line_containing(token)
        end

    %{
      file: file_id,
      line: max(line || 1, 1),
      line_start: max(line || 1, 1),
      line_end: max(line || 1, 1),
      token: token
    }
  end

  def verify_trace!(trace) do
    source_by_id = Map.new(trace.sources, &{&1.id, &1})

    Enum.each(trace.events, fn event ->
      verify_source_ref!(event.source, source_by_id, event)
      verify_source_ref!(event.implementation_source, source_by_id, event)
    end)

    trace
  end

  defp verify_source_ref!(nil, _source_by_id, _event), do: :ok

  defp verify_source_ref!(source_ref, source_by_id, event) do
    source = Map.fetch!(source_by_id, source_ref.file)
    line_count = source.source |> String.split("\n") |> length()
    line_start = Map.get(source_ref, :line_start) || source_ref.line
    line_end = Map.get(source_ref, :line_end) || source_ref.line

    unless line_start >= 1 and line_end >= line_start and line_end <= line_count do
      raise ArgumentError,
            "invalid source ref #{inspect(source_ref)} on event #{inspect(event.id || event.type)}"
    end
  end

  defp library_source(file_id) do
    Enum.find_value(@library_files, fn
      {^file_id, _title, relative} -> File.read!(Path.join(@source_root, relative))
      _ -> nil
    end)
  end

  defp line_containing(source, token) do
    source
    |> String.split("\n")
    |> Enum.with_index(1)
    |> Enum.find_value(1, fn {line, index} ->
      if token == "" or String.contains?(line, token), do: index, else: nil
    end)
  end

  defp op_name(:+), do: "add"
  defp op_name(:-), do: "sub"
  defp op_name(:*), do: "mul"
  defp op_name(:neg), do: "neg"
  defp op_name(:relu), do: "relu"
end
