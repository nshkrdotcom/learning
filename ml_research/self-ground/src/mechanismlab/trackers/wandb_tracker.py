from __future__ import annotations

import importlib.util

from mechanismlab.backends.base import BackendManifest
from mechanismlab.trackers.base import OptionalDependencyUnavailable, unavailable_manifest


class WandBTracker:
    name = "wandb"

    def __init__(self) -> None:
        if importlib.util.find_spec("wandb") is None:
            raise OptionalDependencyUnavailable("wandb")

    @staticmethod
    def manifest() -> BackendManifest:
        if importlib.util.find_spec("wandb") is None:
            return unavailable_manifest("wandb")
        return BackendManifest(name="wandb", kind="tracker", available=True)
