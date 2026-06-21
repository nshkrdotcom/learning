from __future__ import annotations

import json
import os

import pytest


@pytest.mark.integration
def test_real_sae_intervention_optional(tmp_path) -> None:
    release = os.getenv("SELF_GROUND_SAE_RELEASE")
    sae_id = os.getenv("SELF_GROUND_SAE_ID")
    if not release or not sae_id:
        pytest.skip("set SELF_GROUND_SAE_RELEASE and SELF_GROUND_SAE_ID")

    from self_ground.real_sae_intervention import run_real_sae_intervention

    out_dir = tmp_path / "sae_intervention"
    result = run_real_sae_intervention(
        out_dir=out_dir,
        per_family=1,
        top_k_features=2,
        sae_release=release,
        sae_id=sae_id,
        device="cpu",
    )

    assert (out_dir / "compatibility.json").exists()
    compatibility = json.loads((out_dir / "compatibility.json").read_text())
    if result.compatible:
        assert compatibility["compatible"] is True
        rows = [
            json.loads(line)
            for line in (out_dir / "intervention_results.jsonl").read_text().splitlines()
        ]
        assert rows
        assert "delta" in rows[0]
        assert "proxy" not in rows[0]
    else:
        assert compatibility["compatible"] is False
        assert not (out_dir / "intervention_results.jsonl").exists()
