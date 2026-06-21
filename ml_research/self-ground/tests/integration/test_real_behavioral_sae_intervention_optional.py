from __future__ import annotations

import json
import os

import pytest


@pytest.mark.integration
def test_real_behavioral_sae_intervention_optional(tmp_path) -> None:
    release = os.getenv("SELF_GROUND_SAE_RELEASE")
    sae_id = os.getenv("SELF_GROUND_SAE_ID")
    if not release or not sae_id:
        pytest.skip("set SELF_GROUND_SAE_RELEASE and SELF_GROUND_SAE_ID")
    model = os.getenv("SELF_GROUND_SAE_MODEL", "EleutherAI/pythia-70m-deduped")

    from self_ground.real_behavioral_intervention import run_real_behavioral_sae_intervention
    from self_ground.real_ranking import run_activation_ranking

    ranking_dir = tmp_path / "ranking"
    out_dir = tmp_path / "phase3"
    run_activation_ranking(
        out_dir=ranking_dir,
        model_name=model,
        hook_point="blocks.2.hook_resid_post",
        feature_source="sae",
        sae_release=release,
        sae_id=sae_id,
        per_family=1,
        top_k_features=5,
        device="cpu",
    )

    run = run_real_behavioral_sae_intervention(
        out_dir=out_dir,
        ranking_dir=ranking_dir,
        model_name=model,
        hook_point="blocks.2.hook_resid_post",
        sae_release=release,
        sae_id=sae_id,
        per_family=2,
        top_k_features=2,
        baseline_mode="top-vs-random",
        operations=["ablate"],
        patch_mode="delta",
        device="cpu",
    )

    compatibility = json.loads((out_dir / "compatibility.json").read_text())
    assert compatibility["reconstruction_compatible"] is True
    assert (out_dir / "behavioral_task_validation.json").exists()
    if run.compatible and run.task_validation_passed:
        rows = [
            json.loads(line)
            for line in (out_dir / "behavioral_intervention_results.jsonl")
            .read_text()
            .splitlines()
        ]
        assert rows
        assert isinstance(rows[0]["target_signed_delta"], float)
        assert isinstance(rows[0]["control_signed_delta"], float)
        assert (out_dir / "behavioral_summary.csv").exists()
        report = json.loads((out_dir / "mechanism_report.json").read_text())
        assert report["claim_status"] != "strong_candidate_evidence"
