from __future__ import annotations

import csv
import json

from self_ground.data import ExperimentResult
from self_ground.io import (
    read_json,
    read_minimal_pairs,
    write_config,
    write_jsonl,
    write_summary_csv,
)
from self_ground.metrics import build_feature_proxy_effect
from self_ground.negation import generate_negation_pairs


def test_jsonl_round_trip_for_minimal_pairs(tmp_path) -> None:
    pairs = generate_negation_pairs(per_family=2, seed=5)
    path = tmp_path / "pairs.jsonl"

    write_jsonl(pairs, path)
    loaded = read_minimal_pairs(path)

    assert [pair.model_dump() for pair in loaded] == [pair.model_dump() for pair in pairs]


def test_summary_csv_has_stable_columns(tmp_path) -> None:
    effect = build_feature_proxy_effect(
        feature_id="negation",
        delta_pos=1.0,
        delta_neg=0.0,
        delta_para=1.0,
        delta_decoy=0.0,
        delta_amplify_or_patch_toward_target=0.5,
        delta_rescue_after_ablation=0.5,
        collateral=0.0,
        mechanism_size=1,
    )
    result = ExperimentResult(
        pair_id="pair-1",
        feature_id="negation",
        template_family="copula",
        metrics=effect,
        metadata={"adapter": "test-local"},
    )
    path = tmp_path / "summary.csv"

    write_summary_csv([result], path)

    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)

    assert header == [
        "feature_id",
        "n_pairs",
        "proxy_necessity_mean",
        "proxy_sufficiency_mean",
        "proxy_specificity_mean",
        "collateral_proxy_mean",
        "proxy_cleanliness_mean",
    ]


def test_config_is_persisted(tmp_path) -> None:
    path = tmp_path / "config.json"
    config = {"model": "test-local", "layer": "test.layer", "top_k_features": 4}

    write_config(config, path)

    assert read_json(path) == config
    assert json.loads(path.read_text()) == config
