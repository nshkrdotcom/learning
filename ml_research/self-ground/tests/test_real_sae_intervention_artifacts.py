from __future__ import annotations

import csv
import json
from types import SimpleNamespace

from self_ground.activations import FeatureActivations
from self_ground.real_ranking import run_activation_ranking
from self_ground.real_sae_intervention import run_real_sae_intervention


class TinyIdentitySAE:
    def __init__(
        self,
        bad_decode: bool = False,
        *,
        model_name: str = "test-local",
        hook_name: str = "blocks.2.hook_resid_post",
    ) -> None:
        self.bad_decode = bad_decode
        metadata = SimpleNamespace(model_name=model_name, hook_name=hook_name)
        self.sae = SimpleNamespace(
            cfg=SimpleNamespace(
                metadata=metadata,
                d_in=4,
                d_sae=4,
                architecture=lambda: "standard",
            )
        )

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
        hook_point="blocks.2.hook_resid_post",
        feature_source="sae",
        sae_release="test-release",
        sae_id="blocks.2.hook_resid_post",
        top_k_features=2,
        model_adapter=tiny_model_adapter,
        sae_adapter=sae,
    )

    result = run_real_sae_intervention(
        out_dir=out_dir,
        ranking_dir=ranking_dir,
        model_name="test-local",
        hook_point="blocks.2.hook_resid_post",
        sae_release="test-release",
        sae_id="blocks.2.hook_resid_post",
        top_k_features=2,
        operation="ablate",
        patch_mode="delta",
        model_adapter=tiny_model_adapter,
        sae_adapter=sae,
    )

    assert result.compatible is True
    assert (out_dir / "compatibility.json").exists()
    assert (out_dir / "intervention_results.jsonl").exists()
    compatibility = json.loads((out_dir / "compatibility.json").read_text())
    assert compatibility["metadata_compatible"] is True
    assert compatibility["shape_compatible"] is True
    assert compatibility["reconstruction_compatible"] is True
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
        hook_point="blocks.2.hook_resid_post",
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


def test_sae_intervention_blocks_semantic_model_mismatch(
    tmp_path,
    tiny_model_adapter,
) -> None:
    out_dir = tmp_path / "blocked_semantic"

    result = run_real_sae_intervention(
        out_dir=out_dir,
        model_name="EleutherAI/pythia-70m",
        hook_point="blocks.2.hook_resid_post",
        sae_release="pythia-70m-deduped-res-sm",
        sae_id="blocks.2.hook_resid_post",
        top_k_features=2,
        model_adapter=tiny_model_adapter,
        sae_adapter=TinyIdentitySAE(model_name="pythia-70m-deduped"),
    )

    assert result.compatible is False
    assert (out_dir / "config.json").exists()
    assert (out_dir / "compatibility.json").exists()
    assert (out_dir / "README.md").exists()
    assert not (out_dir / "intervention_results.jsonl").exists()
    assert not (out_dir / "summary.csv").exists()
    assert not (out_dir / "selected_features.json").exists()
    compatibility = json.loads((out_dir / "compatibility.json").read_text())
    assert compatibility["shape_compatible"] is True
    assert compatibility["metadata_compatible"] is False
    readme = (out_dir / "README.md").read_text()
    assert "pythia-70m-deduped" in readme
    assert "EleutherAI/pythia-70m" in readme
    assert "different checkpoints" in readme
