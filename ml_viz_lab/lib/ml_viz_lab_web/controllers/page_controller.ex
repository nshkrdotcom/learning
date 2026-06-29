defmodule MlVizLabWeb.PageController do
  use MlVizLabWeb, :controller

  def home(conn, _params) do
    render(conn, :home)
  end
end
