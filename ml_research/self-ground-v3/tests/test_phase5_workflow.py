import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from mwb.cli import app
from mwb.context import RunContext
from mwb.project import ProjectManager
from mwb.session import SessionManager
from mwb.workflows.preflight import run_preflight
from mwb.workflows.verify import run_verify


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# repo\n")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def make_context(tmp_path: Path) -> RunContext:
    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    session = SessionManager.start(project, surface="test")
    return RunContext(project=project, session=session)


def test_hypothesis_create_and_prediction_lock_are_hash_stable(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    bundle = ctx.domains.negation.load("phase3_calibrated")
    h = ctx.hypotheses.create(
        title="candidate negation feature",
        units=["unit_1"],
        example_bundle=bundle.targets,
        control_bundle=bundle.controls,
        expected_effect="target_delta > matched_control_delta",
        required_controls=["negation_removed"],
        alternative_explanations=["control_leaky"],
    )
    lock = ctx.predictions.lock(
        h,
        expected_direction="target_delta_positive_control_delta_low",
        expected_controls={"negation_removed": "low_delta"},
    )

    assert h.control_bundle_ref == bundle.controls.wb_ref
    assert lock.hypothesis_ref == h.wb_ref
    assert lock.hypothesis_spec_hash == h.wb_fingerprint()
    assert lock.wb_fingerprint() == lock.wb_fingerprint()


def test_preflight_fails_without_control_bundle() -> None:
    report = run_preflight(
        {
            "wb_ref": "hyp_missing_controls",
            "title": "bad hypothesis",
            "units": ["unit_1"],
            "example_bundle_ref": "bundle_1",
            "control_bundle_ref": "",
            "expected_effect": "target_delta > controls",
            "required_controls": ["negation_removed"],
            "metadata": {"tensor_space_compatible": True},
        }
    )

    assert report.status == "fail"
    assert "missing_control_bundle" in report.blockers


def test_preflight_warns_on_weak_projection() -> None:
    report = run_preflight(
        {
            "wb_ref": "hyp_projection_warn",
            "title": "weak projection",
            "units": ["unit_1"],
            "example_bundle_ref": "bundle_1",
            "control_bundle_ref": "ctrl_1",
            "expected_effect": "target_delta > controls",
            "required_controls": ["negation_removed"],
            "metadata": {
                "tensor_space_compatible": True,
                "model_sae_hook_match": True,
                "decoder_unembed_projection_score": 0.01,
            },
        }
    )

    assert report.status == "warn"
    assert any(check["name"] == "decoder_unembed_projection" for check in report.checks)


def test_verify_requires_prediction_lock_for_claim_bearing() -> None:
    result = run_verify(
        {
            "wb_ref": "hyp_no_lock",
            "title": "claim-bearing without lock",
            "units": ["unit_1"],
            "example_bundle_ref": "bundle_1",
            "control_bundle_ref": "ctrl_1",
            "expected_effect": "target_delta > controls",
            "required_controls": ["negation_removed"],
        },
        prediction_lock=None,
        diagnostic_only=False,
        dry_run=True,
    )

    assert result.status == "blocked"
    assert result.evidence_posture == "blocked"
    assert result.metadata["blockers"] == ["prediction_lock_missing"]


def test_preflight_and_verify_cli(tmp_path: Path, monkeypatch) -> None:
    init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    ProjectManager.init(tmp_path, name="self-ground")
    hypothesis_file = tmp_path / "hypothesis.json"
    hypothesis_file.write_text(
        json.dumps(
            {
                "wb_ref": "hyp_cli",
                "title": "cli hypothesis",
                "units": ["unit_1"],
                "example_bundle_ref": "bundle_1",
                "control_bundle_ref": "ctrl_1",
                "expected_effect": "target_delta > controls",
                "required_controls": ["negation_removed"],
                "metadata": {"tensor_space_compatible": True, "model_sae_hook_match": True},
            }
        )
    )
    runner = CliRunner()

    preflight = runner.invoke(app, ["preflight", str(hypothesis_file)])
    verify = runner.invoke(app, ["verify", str(hypothesis_file), "--diagnostic-only", "--dry-run"])

    assert preflight.exit_code == 0, preflight.output
    assert verify.exit_code == 0, verify.output
    assert json.loads(preflight.output)["status"] == "pass"
    assert json.loads(verify.output)["evidence_posture"] == "diagnostic_only"
