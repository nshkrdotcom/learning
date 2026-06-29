defmodule MlVizLabWeb.LiveExecutionTest do
  use MlVizLabWeb.ConnCase

  import Phoenix.LiveViewTest

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
end
