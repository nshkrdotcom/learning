from __future__ import annotations

import numpy as np

from self_ground.activations import FeatureActivations, PairFeatureActivations
from self_ground.metrics import MetricWeights, build_feature_proxy_effect


def ablate_features(
    feature_activations: FeatureActivations,
    feature_ids: list[str],
) -> FeatureActivations:
    values = np.array(feature_activations.values, copy=True)
    for feature_id in feature_ids:
        values[..., feature_activations.feature_ids.index(feature_id)] = 0.0
    return FeatureActivations(values=values, feature_ids=list(feature_activations.feature_ids))


def amplify_features(
    feature_activations: FeatureActivations,
    feature_ids: list[str],
    *,
    factor: float,
) -> FeatureActivations:
    values = np.array(feature_activations.values, copy=True)
    for feature_id in feature_ids:
        values[..., feature_activations.feature_ids.index(feature_id)] *= factor
    return FeatureActivations(values=values, feature_ids=list(feature_activations.feature_ids))


def patch_feature(
    target: FeatureActivations,
    source: FeatureActivations,
    feature_ids: list[str],
) -> FeatureActivations:
    if target.values.shape != source.values.shape:
        raise ValueError("target and source feature activations must have the same shape")
    values = np.array(target.values, copy=True)
    for feature_id in feature_ids:
        target_idx = target.feature_ids.index(feature_id)
        source_idx = source.feature_ids.index(feature_id)
        values[..., target_idx] = source.values[..., source_idx]
    return FeatureActivations(values=values, feature_ids=list(target.feature_ids))


def _collateral_proxy(
    *,
    features: PairFeatureActivations,
    pair_id: str,
    feature_id: str,
    sae,
) -> float:
    if not hasattr(sae, "decode"):
        return 0.0

    row_indices = features.row_indices_for_pair(pair_id)
    feature_idx = features.feature_index(feature_id)
    original_values = features.values[row_indices, :]
    ablated_values = np.array(original_values, copy=True)
    ablated_values[:, feature_idx] = 0.0
    original = FeatureActivations(original_values, list(features.feature_ids))
    ablated = FeatureActivations(ablated_values, list(features.feature_ids))

    try:
        original_decoded = np.asarray(sae.decode(original), dtype=float)
        ablated_decoded = np.asarray(sae.decode(ablated), dtype=float)
    except Exception:
        return 0.0

    if original_decoded.shape != ablated_decoded.shape:
        return 0.0
    return float(np.mean(np.abs(original_decoded - ablated_decoded)))


def evaluate_feature_space_proxy(
    features: PairFeatureActivations,
    pair_id: str,
    feature_id: str,
    sae,
    *,
    weights: MetricWeights | None = None,
) -> object:
    pos = features.condition_value(pair_id, "x_pos", feature_id)
    neg = features.condition_value(pair_id, "x_neg", feature_id)
    para = features.condition_value(pair_id, "x_para", feature_id)
    decoy = features.condition_value(pair_id, "x_decoy", feature_id)

    target_level = max(pos, para)
    patch_toward_target = max(0.0, target_level - neg) + max(0.0, target_level - decoy)
    rescue_after_ablation = (pos + para) / 2.0
    collateral = _collateral_proxy(
        features=features,
        pair_id=pair_id,
        feature_id=feature_id,
        sae=sae,
    )

    return build_feature_proxy_effect(
        feature_id=feature_id,
        operation="feature_space_ablation_proxy",
        delta_pos=pos,
        delta_neg=neg,
        delta_para=para,
        delta_decoy=decoy,
        delta_amplify_or_patch_toward_target=patch_toward_target,
        delta_rescue_after_ablation=rescue_after_ablation,
        collateral=collateral,
        mechanism_size=1,
        weights=weights,
    )
