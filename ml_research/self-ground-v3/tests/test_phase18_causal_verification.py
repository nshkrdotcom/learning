import json
import math
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from mwb.cli import app
from mwb.domain.objects import PredictionLock
from mwb.project import ProjectManager


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def static_compiler_payload() -> dict:
    return {
        "tensor_space_ref": "space_resid_post",
        "unembedding_space_ref": "space_resid_post",
        "target_token_ids": [10],
        "foil_token_ids": [20],
        "decoder_vector": [1.0, 0.0],
        "unembedding": {"10": [1.0, 0.0], "20": [0.0, 1.0]},
        "dictionary": {
            "feature_id": "unit_sae_123",
            "decoder_vectors": {
                "unit_sae_123": [1.0, 0.0],
                "unit_clean_neighbor": [0.0, 1.0],
            },
        },
        "activation_density": {"target": 0.1, "control": 0.1, "max_ratio": 1.5},
    }


def base_hypothesis(operations: list[dict], *, policy: dict | None = None) -> dict:
    return {
        "wb_ref": "hyp_causal",
        "title": "causal verification hypothesis",
        "units": ["unit_sae_123"],
        "example_bundle_ref": "bundle_1",
        "control_bundle_ref": "ctrl_1",
        "expected_effect": "target_delta > controls",
        "required_controls": ["negation_removed"],
        "metadata": {"tensor_space_compatible": True, "model_sae_hook_match": True},
        "static_compiler": static_compiler_payload(),
        "verification": {
            "baseline": {
                "target_logit": 2.0,
                "control_logit": 1.0,
                "logits": [2.0, 1.0],
                "activation_norm": 1.0,
            },
            "operations": operations,
            "policy": policy or {},
            "telemetry_thresholds": {
                "kl_drift": 0.25,
                "norm_drift": 0.5,
            },
        },
    }


def operation(
    operation_name: str,
    *,
    target_logit: float = 1.2,
    control_logit: float = 0.9,
    logits: list[float] | None = None,
    activation_norm: float = 0.95,
    coefficient: float = 1.0,
) -> dict:
    return {
        "operation": operation_name,
        "unit_ref": "unit_sae_123",
        "patch_mode": operation_name,
        "patch_source": "matched_control",
        "patch_target": "target_prompt",
        "coefficient": coefficient,
        "intervened": {
            "target_logit": target_logit,
            "control_logit": control_logit,
            "logits": logits or [target_logit, control_logit],
            "activation_norm": activation_norm,
        },
    }


def prediction_lock() -> PredictionLock:
    return PredictionLock(
        wb_ref="lock_causal",
        hypothesis_ref="hyp_causal",
        hypothesis_spec_hash="sha256:test",
        expected_direction="target_delta_positive",
        expected_controls={"negation_removed": "low_delta"},
        git_state={"commit": "abc123"},
        environment={"python": "3.12"},
    )


def test_resample_ablation_writes_receipts_and_metrics(tmp_path: Path) -> None:
    from mwb.causal_verification import CausalVerificationService

    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    service = CausalVerificationService(project)

    run = service.verify_payload(
        base_hypothesis([operation("resample_ablate")]),
        prediction_lock=None,
        diagnostic_only=True,
        dry_run=False,
    )

    run_dir = Path(run.metadata["run_dir"])
    receipts = [
        json.loads(line)
        for line in (run_dir / "intervention_receipts.jsonl").read_text().splitlines()
    ]
    results = [
        json.loads(line)
        for line in (run_dir / "verification_results.jsonl").read_text().splitlines()
    ]

    assert run.evidence_posture == "diagnostic_only"
    assert receipts[0]["operation"] == "resample_ablate"
    assert receipts[0]["backend_executed"] is True
    assert results[0]["metric_results"]["target_delta"] == 0.8
    assert math.isclose(results[0]["metric_results"]["specificity_gap"], 0.7)
    assert (run_dir / "telemetry.jsonl").exists()


def test_noising_and_denoising_have_distinct_receipts(tmp_path: Path) -> None:
    from mwb.causal_verification import CausalVerificationService

    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    run = CausalVerificationService(project).verify_payload(
        base_hypothesis([operation("noising"), operation("denoising")]),
        prediction_lock=None,
        diagnostic_only=True,
        dry_run=False,
    )

    receipts = [
        json.loads(line)
        for line in (Path(run.metadata["run_dir"]) / "intervention_receipts.jsonl")
        .read_text()
        .splitlines()
    ]

    assert [receipt["operation"] for receipt in receipts] == ["noising", "denoising"]
    assert receipts[0]["causal_direction"] == "clean_to_corrupt"
    assert receipts[1]["causal_direction"] == "corrupt_to_clean"


def test_feature_amplification_records_coefficient(tmp_path: Path) -> None:
    from mwb.causal_verification import CausalVerificationService

    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    run = CausalVerificationService(project).verify_payload(
        base_hypothesis([operation("feature_amplify", coefficient=1.75)]),
        prediction_lock=None,
        diagnostic_only=True,
        dry_run=False,
    )

    receipt = json.loads(
        (Path(run.metadata["run_dir"]) / "intervention_receipts.jsonl").read_text().splitlines()[0]
    )

    assert receipt["operation"] == "feature_amplify"
    assert receipt["coefficient"] == 1.75


def test_telemetry_drift_blocks_off_manifold_intervention(tmp_path: Path) -> None:
    from mwb.causal_verification import CausalVerificationService

    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    run = CausalVerificationService(project).verify_payload(
        base_hypothesis(
            [
                operation(
                    "resample_ablate",
                    target_logit=-5.0,
                    control_logit=5.0,
                    logits=[-5.0, 5.0],
                    activation_norm=3.0,
                )
            ]
        ),
        prediction_lock=None,
        diagnostic_only=True,
        dry_run=False,
    )

    telemetry = json.loads(
        (Path(run.metadata["run_dir"]) / "telemetry.jsonl").read_text().splitlines()[0]
    )

    assert "off_manifold_intervention" in run.metadata["blockers"]
    assert telemetry["kl_drift"] > 0.25
    assert telemetry["activation_norm_drift"] > 0.5


def test_zero_ablation_has_diagnostic_claim_ceiling(tmp_path: Path) -> None:
    from mwb.causal_verification import CausalVerificationService

    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    run = CausalVerificationService(project).verify_payload(
        base_hypothesis(
            [operation("zero_ablate")],
            policy={"zero_ablation_claim_ceiling": "diagnostic_only"},
        ),
        prediction_lock=prediction_lock(),
        diagnostic_only=False,
        dry_run=False,
    )

    assert run.evidence_posture == "diagnostic_only"
    assert run.metadata["claim_ceiling"] == "diagnostic_only"
    assert "zero_ablation_claim_ceiling" in run.metadata["blockers"]


def test_verify_cli_writes_causal_artifacts(tmp_path: Path, monkeypatch) -> None:
    init_git_repo(tmp_path)
    ProjectManager.init(tmp_path, name="self-ground")
    monkeypatch.chdir(tmp_path)
    hypothesis_file = tmp_path / "hypothesis.json"
    hypothesis_file.write_text(
        json.dumps(base_hypothesis([operation("resample_ablate")])),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["verify", str(hypothesis_file), "--diagnostic-only"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["evidence_posture"] == "diagnostic_only"
    assert (Path(payload["metadata"]["run_dir"]) / "intervention_receipts.jsonl").exists()
