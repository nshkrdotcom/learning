defmodule MlVizLab.Subjects.Adapter do
  @moduledoc """
  Behaviour for a visualized learning subject.

  Subject adapters own the domain-specific work: lesson definitions, source
  catalogs, trace generation, and concept copy. The Phoenix shell consumes only
  the generic maps returned by this boundary.
  """

  @callback id() :: String.t()
  @callback title() :: String.t()
  @callback description() :: String.t()
  @callback capabilities() :: map()
  @callback lessons() :: [MlVizLab.Subjects.Lesson.t() | map()]
  @callback concepts() :: [map()]
  @callback sources(lesson_id :: String.t()) :: [map()]
  @callback run(lesson_id :: String.t(), opts :: keyword()) ::
              {:ok, MlVizLab.Trace.Run.t()} | {:error, term()}
end
