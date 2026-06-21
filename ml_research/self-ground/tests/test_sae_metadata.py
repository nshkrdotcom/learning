from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from self_ground.sae_metadata import (
    extract_sae_identity_metadata,
    metadata_matches_requested_target,
    normalize_model_name,
    parse_transformerlens_hook_point,
)


class TinyMetadataAdapter:
    def __init__(
        self,
        *,
        model_name: str | None = "pythia-70m-deduped",
        hook_name: str | None = "blocks.2.hook_resid_post",
    ) -> None:
        metadata = SimpleNamespace(model_name=model_name, hook_name=hook_name)
        self.sae = SimpleNamespace(
            cfg=SimpleNamespace(
                metadata=metadata,
                d_in=512,
                d_sae=32768,
                architecture=lambda: "standard",
            )
        )


def test_normalizes_safe_model_prefixes_without_losing_variants() -> None:
    assert normalize_model_name("EleutherAI/pythia-70m") == "pythia-70m"
    assert normalize_model_name("pythia-70m") == "pythia-70m"
    assert normalize_model_name("EleutherAI/pythia-70m-deduped") == "pythia-70m-deduped"
    assert normalize_model_name("pythia-70m-deduped") == "pythia-70m-deduped"
    assert normalize_model_name("EleutherAI/pythia-70m") != normalize_model_name(
        "pythia-70m-deduped"
    )


def test_parse_transformerlens_hook_point() -> None:
    parsed = parse_transformerlens_hook_point("blocks.2.hook_resid_post")

    assert parsed["parse_status"] == "ok"
    assert parsed["layer"] == 2
    assert parsed["hook_type"] == "resid_post"
    assert parsed["canonical_hook_point"] == "blocks.2.hook_resid_post"


def test_parse_transformerlens_hook_point_rejects_malformed() -> None:
    with pytest.raises(ValueError, match="unsupported TransformerLens hook point"):
        parse_transformerlens_hook_point("not-a-hook")


def test_metadata_matches_exact_model_and_hook_target() -> None:
    metadata = extract_sae_identity_metadata(
        sae_adapter=TinyMetadataAdapter(),
        sae_release="release",
        sae_id="blocks.2.hook_resid_post",
    )

    report = metadata_matches_requested_target(
        metadata=metadata,
        requested_model_name="EleutherAI/pythia-70m-deduped",
        requested_hook_point="blocks.2.hook_resid_post",
    )

    assert report["metadata_compatible"] is True
    assert report["model_match"] is True
    assert report["hook_point_match"] is True
    assert report["hook_layer_match"] is True
    assert report["hook_type_match"] is True
    assert not report["errors"]


def test_metadata_rejects_deduped_vs_non_deduped_model() -> None:
    metadata = extract_sae_identity_metadata(
        sae_adapter=TinyMetadataAdapter(model_name="pythia-70m-deduped"),
        sae_release="release",
        sae_id="blocks.2.hook_resid_post",
    )

    report = metadata_matches_requested_target(
        metadata=metadata,
        requested_model_name="EleutherAI/pythia-70m",
        requested_hook_point="blocks.2.hook_resid_post",
    )

    assert report["metadata_compatible"] is False
    assert report["model_match"] is False
    assert "different checkpoints" in " ".join(report["errors"])


def test_metadata_rejects_hook_layer_mismatch() -> None:
    metadata = extract_sae_identity_metadata(
        sae_adapter=TinyMetadataAdapter(hook_name="blocks.2.hook_resid_post"),
        sae_release="release",
        sae_id="blocks.2.hook_resid_post",
    )

    report = metadata_matches_requested_target(
        metadata=metadata,
        requested_model_name="pythia-70m-deduped",
        requested_hook_point="blocks.3.hook_resid_post",
    )

    assert report["metadata_compatible"] is False
    assert report["hook_layer_match"] is False


def test_metadata_rejects_hook_type_mismatch() -> None:
    metadata = extract_sae_identity_metadata(
        sae_adapter=TinyMetadataAdapter(hook_name="blocks.2.hook_resid_post"),
        sae_release="release",
        sae_id="blocks.2.hook_resid_post",
    )

    report = metadata_matches_requested_target(
        metadata=metadata,
        requested_model_name="pythia-70m-deduped",
        requested_hook_point="blocks.2.hook_mlp_out",
    )

    assert report["metadata_compatible"] is False
    assert report["hook_type_match"] is False


def test_metadata_fails_closed_when_required_identity_fields_are_missing() -> None:
    metadata = extract_sae_identity_metadata(
        sae_adapter=TinyMetadataAdapter(model_name=None, hook_name=None),
        sae_release="release",
        sae_id="blocks.2.hook_resid_post",
    )

    report = metadata_matches_requested_target(
        metadata=metadata,
        requested_model_name="pythia-70m-deduped",
        requested_hook_point="blocks.2.hook_resid_post",
    )

    assert report["metadata_compatible"] is False
    assert "declared_model" in report["missing_metadata_fields"]
    assert "declared_hook_point" in report["missing_metadata_fields"]


def test_metadata_report_serializes_cleanly() -> None:
    metadata = extract_sae_identity_metadata(
        sae_adapter=TinyMetadataAdapter(),
        sae_release="release",
        sae_id="blocks.2.hook_resid_post",
    )
    report = metadata_matches_requested_target(
        metadata=metadata,
        requested_model_name="EleutherAI/pythia-70m-deduped",
        requested_hook_point="blocks.2.hook_resid_post",
    )

    encoded = json.dumps({"metadata": metadata.model_dump(), "report": report})

    assert json.loads(encoded)["report"]["metadata_compatible"] is True
