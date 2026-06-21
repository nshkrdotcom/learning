from __future__ import annotations

from self_ground.activations import collect_pair_feature_activations, rank_candidate_features
from self_ground.interventions import evaluate_feature_space_proxy
from self_ground.negation import generate_negation_pairs


def test_candidate_ranking_prefers_negation_over_decoy_and_topic_features(
    tiny_model_adapter,
    tiny_sae_adapter,
) -> None:
    pairs = generate_negation_pairs(per_family=3, seed=11)

    features = collect_pair_feature_activations(
        pairs,
        model=tiny_model_adapter,
        sae=tiny_sae_adapter,
        layer="test.layer",
    )
    ranking = rank_candidate_features(features)
    by_id = {row.feature_id: row.score for row in ranking}

    assert ranking[0].feature_id == "negation"
    assert by_id["negation"] > by_id["frequency"]
    assert by_id["negation"] > by_id["topic"]
    assert by_id["frequency"] < 0.0
    assert abs(by_id["topic"]) < 1e-9


def test_ablation_math_is_selective_for_negation_conditions(
    tiny_model_adapter,
    tiny_sae_adapter,
) -> None:
    pair = generate_negation_pairs(per_family=1, seed=2)[0]
    features = collect_pair_feature_activations(
        [pair],
        model=tiny_model_adapter,
        sae=tiny_sae_adapter,
        layer="test.layer",
    )

    effect = evaluate_feature_space_proxy(
        features=features,
        pair_id=pair.id,
        feature_id="negation",
        sae=tiny_sae_adapter,
    )

    assert effect.delta_pos > 0.9
    assert effect.delta_para > 0.9
    assert abs(effect.delta_neg) < 1e-9
    assert abs(effect.delta_decoy) < 1e-9
    assert effect.proxy_necessity > 1.8
    assert effect.proxy_sufficiency > 1.0
    assert effect.proxy_specificity > 1.8


def test_dirty_feature_has_nonzero_collateral_and_lower_cleanliness(
    tiny_model_adapter,
    tiny_sae_adapter,
) -> None:
    pair = generate_negation_pairs(per_family=1, seed=2)[0]
    features = collect_pair_feature_activations(
        [pair],
        model=tiny_model_adapter,
        sae=tiny_sae_adapter,
        layer="test.layer",
    )

    clean = evaluate_feature_space_proxy(features, pair.id, "negation", tiny_sae_adapter)
    dirty = evaluate_feature_space_proxy(features, pair.id, "dirty_broad", tiny_sae_adapter)

    assert dirty.collateral_proxy > 0.0
    assert dirty.collateral_proxy > clean.collateral_proxy
    assert clean.proxy_cleanliness > dirty.proxy_cleanliness
