from __future__ import annotations

import csv
import json

from self_ground.activations import FeatureActivations
from self_ground.real_ranking import run_activation_ranking
from self_ground.real_sae_intervention import run_real_sae_intervention


class TinyIdentitySAE:
    def __init__(self, bad_decode: bool = False) -> None:
        self.bad_decode = bad_decode

    def encode(self, activation) -> FeatureActivations:
        return FeatureActivations(
            values=activation.detach().cpu().numpy()
            if hasattr(activation, "detach")
            else activation,
            feature_ids=[f"sae_{idx}" for idx in range(activation.shape[-1])],
        )

    def decode(self, feature_activations: FeatureActivations):
        values = feature_activations.values
        if self.bad_decode:
            return values[..., :-1]
        return values


def test_real_sae_intervention_artifacts_with_test_local_adapters(
    tmp_path,
    tiny_model_adapter,
) -> None:
    sae = TinyIdentitySAE()
    ranking_dir = tmp_path / "sae_ranking"
    out_dir = tmp_path / "sae_intervention"
    run_activation_ranking(
        out_dir=ranking_dir,
        per_family=1,
        model_name="test-local",
        hook_point="test.layer",
        feature_source="sae",
        sae_release="test-release",
        sae_id="test-sae",
        top_k_features=2,
        model_adapter=tiny_model_adapter,
        sae_adapter=sae,
    )

    result = run_real_sae_intervention(
        out_dir=out_dir,
        ranking_dir=ranking_dir,
        model_name="test-local",
        hook_point="test.layer",
        sae_release="test-release",
        sae_id="test-sae",
        top_k_features=2,
        operation="ablate",
        patch_mode="delta",
        model_adapter=tiny_model_adapter,
        sae_adapter=sae,
    )

    assert result.compatible is True
    assert (out_dir / "compatibility.json").exists()
    assert (out_dir / "intervention_results.jsonl").exists()
    rows = [
        json.loads(line)
        for line in (out_dir / "intervention_results.jsonl").read_text().splitlines()
    ]
    assert rows
    assert "signed_specificity_score" in rows[0]
    assert "absolute_specificity_score" in rows[0]
    assert "proxy" not in rows[0]

    with (out_dir / "summary.csv").open(newline="") as handle:
        header = next(csv.reader(handle))
    assert header == [
        "feature_set",
        "operation",
        "patch_mode",
        "n_pairs",
        "signed_negation_delta_mean",
        "signed_control_delta_mean",
        "absolute_negation_delta_mean",
        "absolute_control_delta_mean",
        "signed_specificity_score_mean",
        "absolute_specificity_score_mean",
    ]


def test_sae_intervention_writes_blocker_without_fake_rows_on_incompatibility(
    tmp_path,
    tiny_model_adapter,
) -> None:
    out_dir = tmp_path / "blocked"

    result = run_real_sae_intervention(
        out_dir=out_dir,
        model_name="test-local",
        hook_point="test.layer",
        sae_release="test-release",
        sae_id="bad-shape",
        top_k_features=2,
        model_adapter=tiny_model_adapter,
        sae_adapter=TinyIdentitySAE(bad_decode=True),
    )

    assert result.compatible is False
    assert (out_dir / "config.json").exists()
    assert (out_dir / "compatibility.json").exists()
    assert (out_dir / "README.md").exists()
    assert not (out_dir / "intervention_results.jsonl").exists()
    compatibility = json.loads((out_dir / "compatibility.json").read_text())
    assert compatibility["compatible"] is False
    assert compatibility["error"]
