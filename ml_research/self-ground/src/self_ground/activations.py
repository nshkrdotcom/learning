from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from self_ground.data import MinimalPair

CONDITIONS = ("x_pos", "x_neg", "x_para", "x_decoy")


@dataclass(frozen=True)
class FeatureActivations:
    values: np.ndarray
    feature_ids: list[str]

    def __post_init__(self) -> None:
        values = _to_numpy(self.values)
        if values.ndim < 2:
            raise ValueError("feature activations must have at least batch and feature dimensions")
        object.__setattr__(self, "values", values)
        if values.shape[-1] != len(self.feature_ids):
            raise ValueError("last activation dimension must match feature_ids")


@dataclass(frozen=True)
class PairFeatureActivations:
    pair_ids: list[str]
    template_families: list[str]
    conditions: list[str]
    values: np.ndarray
    feature_ids: list[str]

    def feature_index(self, feature_id: str) -> int:
        try:
            return self.feature_ids.index(feature_id)
        except ValueError as exc:
            raise KeyError(f"unknown feature_id: {feature_id}") from exc

    def condition_value(self, pair_id: str, condition: str, feature_id: str) -> float:
        feature_idx = self.feature_index(feature_id)
        for idx, (candidate_pair, candidate_condition) in enumerate(
            zip(self.pair_ids, self.conditions, strict=True)
        ):
            if candidate_pair == pair_id and candidate_condition == condition:
                return float(self.values[idx, feature_idx])
        raise KeyError(f"missing condition {condition!r} for pair {pair_id!r}")

    def row_indices_for_pair(self, pair_id: str) -> list[int]:
        return [idx for idx, candidate in enumerate(self.pair_ids) if candidate == pair_id]

    def template_family_for_pair(self, pair_id: str) -> str:
        for candidate, family in zip(self.pair_ids, self.template_families, strict=True):
            if candidate == pair_id:
                return family
        raise KeyError(f"unknown pair_id: {pair_id}")


@dataclass(frozen=True)
class RankedFeature:
    feature_id: str
    score: float
    mean_pos: float
    mean_neg: float
    mean_para: float
    mean_decoy: float


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _feature_matrix(feature_activations: FeatureActivations) -> np.ndarray:
    values = _to_numpy(feature_activations.values)
    if values.ndim == 2:
        return values
    return values.mean(axis=tuple(range(1, values.ndim - 1)))


def _condition_text(pair: MinimalPair, condition: str) -> str:
    return getattr(pair, condition)


def collect_pair_feature_activations(
    pairs: list[MinimalPair],
    *,
    model,
    sae,
    layer: str,
) -> PairFeatureActivations:
    texts: list[str] = []
    pair_ids: list[str] = []
    template_families: list[str] = []
    conditions: list[str] = []
    for pair in pairs:
        for condition in CONDITIONS:
            texts.append(_condition_text(pair, condition))
            pair_ids.append(pair.id)
            template_families.append(pair.template_family)
            conditions.append(condition)

    activations = model.get_activations(texts, hook_point=layer)
    feature_activations = sae.encode(activations)
    values = _feature_matrix(feature_activations)

    return PairFeatureActivations(
        pair_ids=pair_ids,
        template_families=template_families,
        conditions=conditions,
        values=values,
        feature_ids=list(feature_activations.feature_ids),
    )


def rank_candidate_features(features: PairFeatureActivations) -> list[RankedFeature]:
    rows: list[RankedFeature] = []
    conditions = np.asarray(features.conditions)
    for feature_idx, feature_id in enumerate(features.feature_ids):
        column = features.values[:, feature_idx]
        mean_pos = float(column[conditions == "x_pos"].mean())
        mean_neg = float(column[conditions == "x_neg"].mean())
        mean_para = float(column[conditions == "x_para"].mean())
        mean_decoy = float(column[conditions == "x_decoy"].mean())
        score = mean_pos + mean_para - mean_neg - mean_decoy
        rows.append(
            RankedFeature(
                feature_id=feature_id,
                score=float(score),
                mean_pos=mean_pos,
                mean_neg=mean_neg,
                mean_para=mean_para,
                mean_decoy=mean_decoy,
            )
        )
    return sorted(rows, key=lambda row: (-row.score, row.feature_id))
