from __future__ import annotations

import json

import pytest


@pytest.mark.integration
def test_real_residual_intervention_outputs_nonzero_delta(tmp_path) -> None:
    from self_ground.real_ranking import run_activation_ranking
    from self_ground.real_residual_intervention import run_real_residual_intervention

    ranking_dir = tmp_path / "ranking"
    intervention_dir = tmp_path / "intervention"
    run_activation_ranking(
        out_dir=ranking_dir,
        per_family=1,
        top_k_features=2,
        model_name="EleutherAI/pythia-70m",
        hook_point="blocks.2.hook_resid_post",
        feature_source="residual_dimensions",
        device="cpu",
    )
    run_real_residual_intervention(
        out_dir=intervention_dir,
        ranking_dir=ranking_dir,
        top_k_features=2,
        model_name="EleutherAI/pythia-70m",
        hook_point="blocks.2.hook_resid_post",
        device="cpu",
    )

    assert (intervention_dir / "config.json").exists()
    assert (intervention_dir / "selected_features.json").exists()
    assert (intervention_dir / "intervention_results.jsonl").exists()
    rows = [
        json.loads(line)
        for line in (intervention_dir / "intervention_results.jsonl").read_text().splitlines()
    ]
    assert rows
    assert any(
        abs(value) > 0.0
        for row in rows
        for value in row["delta"].values()
    )
