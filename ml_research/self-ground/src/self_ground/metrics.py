from __future__ import annotations

from dataclasses import dataclass

from self_ground.data import FeatureProxyEffect


@dataclass(frozen=True)
class MetricWeights:
    alpha_necessity: float = 1.0
    alpha_sufficiency: float = 1.0
    alpha_specificity: float = 1.0
    beta_collateral: float = 1.0
    beta_size: float = 0.05


def compute_necessity(
    delta_ablate_pos: float,
    delta_ablate_para: float,
    delta_ablate_neg: float,
    delta_ablate_decoy: float,
) -> float:
    return (delta_ablate_pos + delta_ablate_para) - (
        delta_ablate_neg + delta_ablate_decoy
    )


def compute_proxy_necessity(
    delta_ablate_pos: float,
    delta_ablate_para: float,
    delta_ablate_neg: float,
    delta_ablate_decoy: float,
) -> float:
    return compute_necessity(
        delta_ablate_pos,
        delta_ablate_para,
        delta_ablate_neg,
        delta_ablate_decoy,
    )


def compute_sufficiency(
    delta_amplify_or_patch_toward_target: float,
    delta_rescue_after_ablation: float,
    collateral_shift: float,
) -> float:
    return delta_amplify_or_patch_toward_target + delta_rescue_after_ablation - collateral_shift


def compute_proxy_sufficiency(
    delta_amplify_or_patch_toward_target: float,
    delta_rescue_after_ablation: float,
    collateral_shift: float,
) -> float:
    return compute_sufficiency(
        delta_amplify_or_patch_toward_target,
        delta_rescue_after_ablation,
        collateral_shift,
    )


def compute_specificity(
    delta_pos: float,
    delta_para: float,
    delta_neg: float,
    delta_decoy: float,
) -> float:
    return delta_pos + delta_para - delta_neg - delta_decoy


def compute_proxy_specificity(
    delta_pos: float,
    delta_para: float,
    delta_neg: float,
    delta_decoy: float,
) -> float:
    return compute_specificity(delta_pos, delta_para, delta_neg, delta_decoy)


def compute_cleanliness(
    *,
    necessity: float,
    sufficiency: float,
    specificity: float,
    collateral: float,
    mechanism_size: int,
    weights: MetricWeights | None = None,
) -> float:
    w = weights or MetricWeights()
    return (
        (w.alpha_necessity * necessity)
        + (w.alpha_sufficiency * sufficiency)
        + (w.alpha_specificity * specificity)
        - (w.beta_collateral * collateral)
        - (w.beta_size * mechanism_size)
    )


def compute_proxy_cleanliness(
    *,
    proxy_necessity: float,
    proxy_sufficiency: float,
    proxy_specificity: float,
    collateral_proxy: float,
    mechanism_size: int,
    weights: MetricWeights | None = None,
) -> float:
    return compute_cleanliness(
        necessity=proxy_necessity,
        sufficiency=proxy_sufficiency,
        specificity=proxy_specificity,
        collateral=collateral_proxy,
        mechanism_size=mechanism_size,
        weights=weights,
    )


def build_feature_proxy_effect(
    *,
    feature_id: str,
    delta_pos: float,
    delta_neg: float,
    delta_para: float,
    delta_decoy: float,
    delta_amplify_or_patch_toward_target: float,
    delta_rescue_after_ablation: float,
    collateral: float,
    mechanism_size: int,
    operation: str = "feature_space_ablation_proxy",
    weights: MetricWeights | None = None,
) -> FeatureProxyEffect:
    proxy_necessity = compute_proxy_necessity(delta_pos, delta_para, delta_neg, delta_decoy)
    proxy_sufficiency = compute_proxy_sufficiency(
        delta_amplify_or_patch_toward_target,
        delta_rescue_after_ablation,
        collateral,
    )
    proxy_specificity = compute_proxy_specificity(delta_pos, delta_para, delta_neg, delta_decoy)
    proxy_cleanliness = compute_proxy_cleanliness(
        proxy_necessity=proxy_necessity,
        proxy_sufficiency=proxy_sufficiency,
        proxy_specificity=proxy_specificity,
        collateral_proxy=collateral,
        mechanism_size=mechanism_size,
        weights=weights,
    )
    return FeatureProxyEffect(
        feature_id=feature_id,
        operation=operation,  # type: ignore[arg-type]
        delta_pos=delta_pos,
        delta_neg=delta_neg,
        delta_para=delta_para,
        delta_decoy=delta_decoy,
        proxy_necessity=proxy_necessity,
        proxy_sufficiency=proxy_sufficiency,
        proxy_specificity=proxy_specificity,
        collateral_proxy=collateral,
        proxy_cleanliness=proxy_cleanliness,
    )
