defmodule MlVizLabWeb.VizLiveTest do
  use MlVizLabWeb.ConnCase

  alias MlVizLab.Runs
  alias MlVizLabWeb.VizLive

  test "GET / renders the visualization shell", %{conn: conn} do
    conn = get(conn, ~p"/")
    html = html_response(conn, 200)

    assert html =~ ~s(id="viz-lab")
    assert html =~ "data-trace="
    assert html =~ "data-subjects="
    assert html =~ "MicrogradEx"
    assert html =~ "ML Viz Lab"
  end

  test "stale async trace results are ignored" do
    stale_trace = Runs.generate("micrograd", "x_squared", "old-run")

    socket = %Phoenix.LiveView.Socket{
      assigns: %{
        run_id: "new-run",
        subject_id: "micrograd",
        lesson_id: "repeated_parent"
      }
    }

    assert {:noreply, returned_socket} =
             VizLive.handle_async({:trace, "old-run"}, {:ok, stale_trace}, socket)

    assert returned_socket.assigns.run_id == "new-run"
    refute Map.has_key?(returned_socket.assigns, :trace)
  end
end
