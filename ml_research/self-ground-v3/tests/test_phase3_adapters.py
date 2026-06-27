import json

from typer.testing import CliRunner

from mwb.adapters.conformance import claim_bearing_gate
from mwb.adapters.manifests import AdapterConformanceResult
from mwb.adapters.saelens import SAELensAdapter
from mwb.adapters.transformer_lens import TransformerLensAdapter
from mwb.cli import app


def test_transformer_lens_manifest_declares_p0_capabilities() -> None:
    manifest = TransformerLensAdapter().capability_manifest()

    assert manifest.adapter_name == "TransformerLensAdapter"
    assert manifest.capabilities["load_model"] is True
    assert manifest.capabilities["capture_activation"] is True
    assert manifest.claim_bearing.supported is True
    assert "activation_capture_roundtrip" in manifest.claim_bearing.required_conformance


def test_saelens_manifest_declares_p0_capabilities() -> None:
    manifest = SAELensAdapter().capability_manifest()

    assert manifest.adapter_name == "SAELensAdapter"
    assert manifest.capabilities["load_sae"] is True
    assert manifest.capabilities["encode_activations"] is True
    assert manifest.claim_bearing.supported is True
    assert "sae_identity_roundtrip" in manifest.claim_bearing.required_conformance


def test_backend_version_manifest_uses_installed_packages() -> None:
    manifest = TransformerLensAdapter().backend_version_manifest(device="cpu")

    assert manifest.package_versions["transformer_lens"]
    assert manifest.package_versions["torch"]
    assert manifest.device == "cpu"


def test_transformer_lens_identity_and_tensor_space_are_stable() -> None:
    adapter = TransformerLensAdapter()
    identity = adapter.model_identity_for_name("EleutherAI/pythia-70m-deduped")
    space = adapter.tensor_space_for_hook(
        model_ref=identity.wb_ref,
        hook_point="blocks.2.hook_resid_post",
        d_model=512,
    )

    assert identity.wb_ref.startswith("model_")
    assert identity.model_name == "EleutherAI/pythia-70m-deduped"
    assert space.hook_point == "blocks.2.hook_resid_post"
    assert space.shape == [None, None, 512]


def test_claim_bearing_gate_blocks_missing_saelens() -> None:
    tl_result = AdapterConformanceResult(
        adapter_name="TransformerLensAdapter",
        status="pass",
        checks=[{"name": "load_model_identity", "status": "pass"}],
    )

    gate = claim_bearing_gate(
        [tl_result],
        required_adapters=["TransformerLensAdapter", "SAELensAdapter"],
        required_refs={
            "model_ref": "model_1",
            "dictionary_ref": "dict_1",
            "tensor_space_ref": "space_1",
            "example_bundle_ref": "bundle_1",
            "control_bundle_ref": "ctrl_1",
            "prediction_lock_ref": "lock_1",
            "backend_version_ref": "backend_1",
        },
    )

    assert gate.supported is False
    assert "missing adapter conformance: SAELensAdapter" in gate.blockers


def test_adapter_conformance_dry_run_cli_outputs_manifest() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "adapter",
            "conformance",
            "transformer-lens",
            "--model",
            "EleutherAI/pythia-70m-deduped",
            "--device",
            "cpu",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["adapter_name"] == "TransformerLensAdapter"
    assert payload["status"] == "diagnostic_only"

