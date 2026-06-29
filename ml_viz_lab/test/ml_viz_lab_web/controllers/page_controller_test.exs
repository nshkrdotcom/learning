defmodule MlVizLabWeb.VizLiveTest do
  use MlVizLabWeb.ConnCase

  test "GET / renders the visualization shell", %{conn: conn} do
    conn = get(conn, ~p"/")
    html = html_response(conn, 200)

    assert html =~ ~s(id="viz-lab")
    assert html =~ "data-trace="
    assert html =~ "ML Viz Lab"
  end
end
