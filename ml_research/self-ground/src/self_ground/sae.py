from __future__ import annotations

import numpy as np
import torch

from self_ground.activations import FeatureActivations


class SAELensAdapter:
    """Thin real SAE adapter around SAELens.

    Loading is intentionally explicit: callers must provide the SAELens release
    and SAE id for the pretrained artifact they want to evaluate.
    """

    def __init__(self, sae, *, device: str = "cpu") -> None:
        self.sae = sae
        self.device = device
        if hasattr(self.sae, "eval"):
            self.sae.eval()
        cfg = getattr(self.sae, "cfg", None)
        self.d_in = int(getattr(cfg, "d_in", 0) or getattr(self.sae, "d_in", 0) or 0)
        self.d_sae = int(getattr(cfg, "d_sae", 0) or getattr(self.sae, "d_sae", 0) or 0)

    @classmethod
    def from_pretrained(cls, *, release: str, sae_id: str, device: str = "cpu") -> SAELensAdapter:
        from sae_lens import SAE

        loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
        sae = loaded[0] if isinstance(loaded, tuple) else loaded
        return cls(sae, device=device)

    def _to_tensor(self, activations) -> torch.Tensor:
        if isinstance(activations, torch.Tensor):
            return activations.to(self.device)
        return torch.as_tensor(np.asarray(activations), dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def encode(self, activations) -> FeatureActivations:
        tensor = self._to_tensor(activations)
        encoded = self.sae.encode(tensor)
        values = encoded.detach().cpu().numpy()
        feature_count = values.shape[-1]
        feature_ids = [f"sae_{idx}" for idx in range(feature_count)]
        return FeatureActivations(values=values, feature_ids=feature_ids)

    @torch.no_grad()
    def decode(self, feature_activations: FeatureActivations) -> np.ndarray:
        tensor = self._to_tensor(feature_activations.values)
        decoded = self.sae.decode(tensor)
        return decoded.detach().cpu().numpy()

    def ablate(
        self,
        feature_activations: FeatureActivations,
        feature_ids: list[str],
    ) -> FeatureActivations:
        values = np.array(feature_activations.values, copy=True)
        for feature_id in feature_ids:
            values[..., feature_activations.feature_ids.index(feature_id)] = 0.0
        return FeatureActivations(values=values, feature_ids=list(feature_activations.feature_ids))

    def amplify(
        self,
        feature_activations: FeatureActivations,
        feature_ids: list[str],
        factor: float,
    ) -> FeatureActivations:
        values = np.array(feature_activations.values, copy=True)
        for feature_id in feature_ids:
            values[..., feature_activations.feature_ids.index(feature_id)] *= factor
        return FeatureActivations(values=values, feature_ids=list(feature_activations.feature_ids))
