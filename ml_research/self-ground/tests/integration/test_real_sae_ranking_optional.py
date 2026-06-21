from __future__ import annotations

import csv
import json
import os

import pytest


@pytest.mark.integration
def test_real_sae_ranking_optional(tmp_path) -> None:
    release = os.getenv("SELF_GROUND_SAE_RELEASE")
    sae_id = os.getenv("SELF_GROUND_SAE_ID")
    model = os.getenv("SELF_GROUND_SAE_MODEL", "EleutherAI/pythia-70m-deduped")
    if not release or not sae_id:
        pytest.skip("set SELF_GROUND_SAE_RELEASE and SELF_GROUND_SAE_ID")

    from self_ground.real_ranking import run_activation_ranking

    out_dir = tmp_path / "sae_ranking"
    run_activation_ranking(
        out_dir=out_dir,
        per_family=1,
        top_k_features=2,
        model_name=model,
        feature_source="sae",
        sae_release=release,
        sae_id=sae_id,
        device="cpu",
    )

    metadata = json.loads((out_dir / "activation_metadata.json").read_text())
    assert metadata["sae_release"] == release
    assert metadata["sae_id"] == sae_id
    assert metadata["metadata_compatible"] is True
    assert metadata["shape_compatible"] is True
    assert metadata["reconstruction_compatible"] is True
    assert metadata["declared_model"]
    assert metadata["declared_hook_point"]
    with (out_dir / "feature_rankings.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["feature_id"].startswith("sae_")
