from __future__ import annotations

import pytest

from self_ground.metrics import (
    MetricWeights,
    build_feature_proxy_effect,
    compute_necessity,
    compute_sufficiency,
)


def test_necessity_and_sufficiency_are_not_conflated() -> None:
    necessity = compute_necessity(
        delta_ablate_pos=1.0,
        delta_ablate_para=0.9,
        delta_ablate_neg=0.1,
        delta_ablate_decoy=0.2,
    )
    sufficiency = compute_sufficiency(
        delta_amplify_or_patch_toward_target=0.7,
        delta_rescue_after_ablation=0.4,
        collateral_shift=0.25,
    )

    assert necessity == pytest.approx(1.6)
    assert sufficiency == pytest.approx(0.85)
    assert necessity != sufficiency


def test_dirty_broad_feature_gets_lower_cleanliness_than_targeted_feature() -> None:
    clean = build_feature_proxy_effect(
        feature_id="negation",
        delta_pos=1.0,
        delta_neg=0.0,
        delta_para=0.95,
        delta_decoy=0.05,
        delta_amplify_or_patch_toward_target=0.9,
        delta_rescue_after_ablation=0.8,
        collateral=0.0,
        mechanism_size=1,
    )
    dirty = build_feature_proxy_effect(
        feature_id="dirty_broad",
        delta_pos=1.0,
        delta_neg=0.0,
        delta_para=0.95,
        delta_decoy=0.05,
        delta_amplify_or_patch_toward_target=0.9,
        delta_rescue_after_ablation=0.8,
        collateral=1.2,
        mechanism_size=1,
    )

    assert clean.proxy_cleanliness > dirty.proxy_cleanliness
    assert dirty.collateral_proxy > clean.collateral_proxy


def test_mechanism_size_penalty_works() -> None:
    weights = MetricWeights(beta_size=0.5)
    small = build_feature_proxy_effect(
        feature_id="small",
        delta_pos=1.0,
        delta_neg=0.0,
        delta_para=1.0,
        delta_decoy=0.0,
        delta_amplify_or_patch_toward_target=0.6,
        delta_rescue_after_ablation=0.6,
        collateral=0.0,
        mechanism_size=1,
        weights=weights,
    )
    large = build_feature_proxy_effect(
        feature_id="large",
        delta_pos=1.0,
        delta_neg=0.0,
        delta_para=1.0,
        delta_decoy=0.0,
        delta_amplify_or_patch_toward_target=0.6,
        delta_rescue_after_ablation=0.6,
        collateral=0.0,
        mechanism_size=4,
        weights=weights,
    )

    assert small.proxy_cleanliness > large.proxy_cleanliness
