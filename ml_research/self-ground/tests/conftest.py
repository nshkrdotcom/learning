from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from self_ground.activations import FeatureActivations
from self_ground.negation import contains_negation_marker


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run slow model/SAE integration tests",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-integration"):
        return
    skip_integration = pytest.mark.skip(reason="use --run-integration to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@dataclass
class TinyActivationBatch:
    texts: list[str]
    activations: np.ndarray
    layer: str


class TinyModelAdapter:
    """Test-local deterministic adapter; not shipped in self_ground.src."""

    def get_activations(self, texts: list[str], hook_point: str) -> np.ndarray:
        rows = []
        for text in texts:
            lowered = text.lower()
            is_negation = float(contains_negation_marker(text))
            is_frequency = float(" often " in lowered or " sometimes " in lowered)
            rows.append(
                [
                    is_negation,
                    is_frequency,
                    1.0,
                    (0.7 * is_negation) + 0.6,
                ]
            )
        return np.asarray(rows, dtype=float)

    def score_negation_behavior(self, text: str) -> float:
        return float(contains_negation_marker(text))


class TinySAEAdapter:
    feature_ids = ["negation", "frequency", "topic", "dirty_broad"]

    def encode(self, activations: np.ndarray) -> FeatureActivations:
        return FeatureActivations(
            values=np.asarray(activations, dtype=float),
            feature_ids=list(self.feature_ids),
        )

    def decode(self, feature_activations: FeatureActivations) -> np.ndarray:
        values = np.asarray(feature_activations.values, dtype=float)
        decoded = np.zeros((values.shape[0], 2), dtype=float)
        decoded[:, 0] = values[:, 0]
        decoded[:, 1] = values[:, 3] * 2.0
        return decoded


@pytest.fixture
def tiny_model_adapter() -> TinyModelAdapter:
    return TinyModelAdapter()


@pytest.fixture
def tiny_sae_adapter() -> TinySAEAdapter:
    return TinySAEAdapter()
