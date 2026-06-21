from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

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

    def __init__(self) -> None:
        self.model = TinyHookedModel(self)

    def _feature_row(self, text: str) -> list[float]:
        lowered = text.lower()
        is_negation = float(contains_negation_marker(text))
        is_frequency = float(" often " in lowered or " sometimes " in lowered)
        return [
            is_negation,
            is_frequency,
            1.0,
            (0.7 * is_negation) + 0.6,
        ]

    def get_activations(self, texts: list[str], hook_point: str) -> np.ndarray:
        return np.asarray([[self._feature_row(text)] for text in texts], dtype=float)

    def logits_for_texts(self, texts: list[str]) -> torch.Tensor:
        activations = torch.tensor(
            [[self._feature_row(text)] for text in texts],
            dtype=torch.float32,
        )
        return self._logits_from_activations(activations)

    def token_ids_for_strings(self, strings: list[str]) -> list[int]:
        ids = []
        for token in strings:
            if token in {" not", " no", " never"}:
                ids.append(0)
            elif token in {" often", " always", " sometimes"}:
                ids.append(1)
            else:
                raise ValueError(f"unknown test token: {token}")
        return ids

    def logit_contrast(
        self,
        texts: list[str],
        positive: list[str],
        negative: list[str],
    ) -> torch.Tensor:
        logits = self.logits_for_texts(texts)[:, -1, :]
        pos_ids = self.token_ids_for_strings(positive)
        neg_ids = self.token_ids_for_strings(negative)
        return logits[:, pos_ids].mean(dim=-1) - logits[:, neg_ids].mean(dim=-1)

    def _logits_from_activations(self, activations: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((activations.shape[0], 1, 6), dtype=torch.float32)
        logits[:, 0, 0] = activations[:, -1, 0]
        logits[:, 0, 1] = activations[:, -1, 1]
        return logits

    def score_negation_behavior(self, text: str) -> float:
        return float(contains_negation_marker(text))


class TinyHookedModel:
    def __init__(self, adapter: TinyModelAdapter) -> None:
        self.adapter = adapter

    def run_with_hooks(self, texts: list[str], fwd_hooks):
        activations = torch.tensor(
            [[self.adapter._feature_row(text)] for text in texts],
            dtype=torch.float32,
        )
        for _, hook_fn in fwd_hooks:
            activations = hook_fn(activations, hook=None)
        return self.adapter._logits_from_activations(activations)


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
