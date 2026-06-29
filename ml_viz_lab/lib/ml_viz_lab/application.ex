defmodule MlVizLab.Application do
  # See https://elixir.hexdocs.pm/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      MlVizLabWeb.Telemetry,
      {DNSCluster, query: Application.get_env(:ml_viz_lab, :dns_cluster_query) || :ignore},
      {Phoenix.PubSub, name: MlVizLab.PubSub},
      MlVizLab.Execution.SessionSupervisor,
      # Start a worker by calling: MlVizLab.Worker.start_link(arg)
      # {MlVizLab.Worker, arg},
      # Start to serve requests, typically the last entry
      MlVizLabWeb.Endpoint
    ]

    # See https://elixir.hexdocs.pm/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: MlVizLab.Supervisor]
    Supervisor.start_link(children, opts)
  end

  # Tell Phoenix to update the endpoint configuration
  # whenever the application is updated.
  @impl true
  def config_change(changed, _new, removed) do
    MlVizLabWeb.Endpoint.config_change(changed, removed)
    :ok
  end
end
