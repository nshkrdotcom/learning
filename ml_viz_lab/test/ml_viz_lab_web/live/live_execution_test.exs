defmodule MlVizLabWeb.LiveExecutionTest do
  use MlVizLabWeb.ConnCase

  import Phoenix.LiveViewTest

  test "continue without a live session emits a structured error and preserves source", %{
    conn: conn
  } do
    {:ok, view, _html} = live(conn, ~p"/?subject=micrograd&lesson=x_squared&mode=live")

    assert live_source(view) =~ "MicrogradEx.Value.new"

    render_hook(view, "continue_live", %{})

    assert_push_event(view, "execution_event", %{
      "type" => "command_error",
      "command" => "continue",
      "reason" => "no_live_session",
      "status" => "idle",
      "session_id" => nil,
      "error" => %{"message" => "Start live execution before continuing."}
    })

    assert live_source(view) =~ "MicrogradEx.Value.new"
  end

  test "step without a live session emits a structured error and preserves source", %{conn: conn} do
    {:ok, view, _html} = live(conn, ~p"/?subject=micrograd&lesson=x_squared&mode=live")

    assert live_source(view) =~ "MicrogradEx.Value.pow"

    render_hook(view, "step_live", %{})

    assert_push_event(view, "execution_event", %{
      "type" => "command_error",
      "command" => "step",
      "reason" => "no_live_session",
      "status" => "idle",
      "session_id" => nil,
      "error" => %{"message" => "Start live execution before stepping."}
    })

    assert live_source(view) =~ "MicrogradEx.Value.pow"
  end

  test "reset pushes an idle live event and preserves source", %{conn: conn} do
    {:ok, view, _html} = live(conn, ~p"/?subject=micrograd&lesson=x_squared&mode=live")

    render_hook(view, "start_live", %{})
    assert_push_event(view, "execution_event", %{"type" => "session_requested"})
    assert_push_event(view, "execution_event", %{"type" => "session_started"})
    assert_push_event(view, "execution_event", %{"type" => "runtime_started"})
    assert_push_event(view, "execution_event", %{"type" => "paused"})

    render_hook(view, "reset_live", %{})

    assert_push_event(view, "execution_event", %{
      "type" => "reset",
      "status" => "idle",
      "session_id" => nil,
      "source" => %{"source" => source}
    })

    assert source =~ "MicrogradEx.Value.new"
    assert live_source(view) =~ "MicrogradEx.Value.new"
  end

  test "live x_squared advances only when backend step commands are sent", %{conn: conn} do
    {:ok, view, html} = live(conn, ~p"/?subject=micrograd&lesson=x_squared&mode=live")

    assert html =~ ~s(data-execution-mode="live")
    assert html =~ "Start live"

    render_hook(view, "start_live", %{})

    assert_push_event(view, "execution_event", %{
      "type" => "session_requested",
      "status" => "starting"
    })

    assert_push_event(view, "execution_event", %{"type" => "session_started"})
    assert_push_event(view, "execution_event", %{"type" => "runtime_started"})

    assert_push_event(view, "execution_event", %{
      "type" => "paused",
      "status" => "paused",
      "span" => %{"line_start" => 1},
      "bindings" => bindings
    })

    refute Map.has_key?(bindings, "x")

    render_hook(view, "step_live", %{})

    assert_push_event(view, "execution_event", %{"type" => "command_sent", "command" => "step"})
    assert_push_event(view, "execution_event", %{"type" => "resumed"})

    assert_push_event(view, "execution_event", %{
      "type" => "binding_snapshot",
      "current_step" => 1,
      "bindings" => bindings
    })

    assert bindings["x"]["domain"] == "micrograd_value"
    refute Map.has_key?(bindings, "y")

    assert_push_event(view, "execution_event", %{
      "type" => "paused",
      "span" => %{"line_start" => 2}
    })

    render_hook(view, "step_live", %{})
    assert_push_event(view, "execution_event", %{"type" => "command_sent", "command" => "step"})
    assert_push_event(view, "execution_event", %{"type" => "resumed"})

    assert_push_event(view, "execution_event", %{
      "type" => "binding_snapshot",
      "current_step" => 2,
      "bindings" => bindings
    })

    assert bindings["y"]["data"] == 9.0

    render_hook(view, "step_live", %{})
    assert_push_event(view, "execution_event", %{"type" => "command_sent", "command" => "step"})
    assert_push_event(view, "execution_event", %{"type" => "resumed"})

    assert_push_event(view, "execution_event", %{
      "type" => "binding_snapshot",
      "current_step" => 3,
      "bindings" => bindings,
      "domain_snapshot" => domain
    })

    assert bindings["gradients"]["domain"] == "micrograd_gradients"
    assert Enum.any?(domain["values"], &(&1["name"] == "x" and &1["gradient"] == 6.0))
    assert_push_event(view, "execution_event", %{"type" => "completed", "status" => "completed"})
  end

  defp live_source(view) do
    view
    |> render()
    |> LazyHTML.from_fragment()
    |> LazyHTML.query("#viz-lab")
    |> LazyHTML.attribute("data-live-source")
    |> List.first()
    |> Jason.decode!()
    |> Map.fetch!("source")
  end
end
