from __future__ import annotations

import importlib.util

from mechanismlab.backends.base import BackendManifest
from mechanismlab.trackers.base import OptionalDependencyUnavailable, unavailable_manifest


class MLflowTracker:
    name = "mlflow"

    def __init__(self) -> None:
        if importlib.util.find_spec("mlflow") is None:
            raise OptionalDependencyUnavailable("mlflow")

    @staticmethod
    def manifest() -> BackendManifest:
        if importlib.util.find_spec("mlflow") is None:
            return unavailable_manifest("mlflow")
        return BackendManifest(name="mlflow", kind="tracker", available=True)
