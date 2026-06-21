from __future__ import annotations

import self_ground.data as data
from self_ground.data import FeatureProxyEffect
from self_ground.experiment import run_negation_experiment
from self_ground.io import write_jsonl
from self_ground.negation import generate_negation_pairs


def test_proxy_schema_uses_proxy_field_names() -> None:
    effect = FeatureProxyEffect(
        feature_id="resid_0",
        operation="feature_space_ablation_proxy",
        delta_pos=1.0,
        delta_neg=0.0,
        delta_para=1.0,
        delta_decoy=0.0,
        proxy_necessity=2.0,
        proxy_sufficiency=1.0,
        proxy_specificity=2.0,
        collateral_proxy=0.0,
        proxy_cleanliness=4.95,
    )

    dumped = effect.model_dump()
    assert "proxy_necessity" in dumped
    assert "proxy_cleanliness" in dumped
    assert "necessity" not in dumped
    assert "cleanliness" not in dumped


def test_no_stale_non_proxy_feature_effect_schema_exists() -> None:
    assert not hasattr(data, "FeatureEffect")


def test_proxy_experiment_writes_proxy_filename_only(
    tmp_path,
    tiny_model_adapter,
    tiny_sae_adapter,
) -> None:
    pairs = generate_negation_pairs(per_family=1, seed=3)
    pairs_path = tmp_path / "pairs.jsonl"
    out_dir = tmp_path / "run"
    write_jsonl(pairs, pairs_path)

    run_negation_experiment(
        pairs_path=pairs_path,
        out_dir=out_dir,
        model_name="test-local",
        layer="test.layer",
        top_k_features=1,
        model_adapter=tiny_model_adapter,
        sae_adapter=tiny_sae_adapter,
    )

    assert (out_dir / "feature_space_proxy_results.jsonl").exists()
    assert not (out_dir / "intervention_results.jsonl").exists()
