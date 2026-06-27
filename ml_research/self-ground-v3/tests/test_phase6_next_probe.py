import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from mwb.cli import app
from mwb.project import ProjectManager
from mwb.workflows.blockers import diagnose_blockers
from mwb.workflows.next_probe import build_next_probe
from mwb.workflows.sweep import parse_axes


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# repo\n")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def test_parse_axes_preserves_cross_product_semantics() -> None:
    parsed = parse_axes(["layer=0,1", "patch_mode=direct", "operation=ablate,amplify"])

    assert parsed["axes"]["layer"] == ["0", "1"]
    assert parsed["matrix_semantics"] == "cross_product"
    assert parsed["matrix_size"] == 4


def test_diagnose_blockers_detects_control_leaky() -> None:
    report = diagnose_blockers(
        {
            "target_delta": 0.5,
            "matched_control_delta": 0.45,
            "specificity_gap": 0.05,
            "family_min_gap": -0.02,
        },
        thresholds={"control_leaky_ratio": 0.8},
    )

    assert report["primary_blocker"] == "control_leaky"
    assert "control_leaky" in report["blockers"]


def test_next_probe_recommends_untried_axis() -> None:
    plan = build_next_probe(
        {
            "run_ref": "run_1",
            "status": "insufficient_evidence",
            "metrics": {
                "target_delta": 0.5,
                "matched_control_delta": 0.45,
                "family_min_gap": -0.02,
            },
            "blockers": ["control_leaky"],
            "tried_axes": {"layers": ["1", "2"], "patch_modes": ["delta"]},
            "available_axes": {"layers": ["0", "1", "2"], "patch_modes": ["delta", "direct"]},
            "backend_capabilities": {"direct_patch": True},
        }
    )

    assert plan.diagnosis["primary"] == "control_leaky"
    assert plan.recommendation["kind"] in {"refresh_controls", "smallest_axis_extension"}
    assert "layer=1" not in plan.recommendation.get("command", "")


def test_next_probe_missing_fields_is_artifact_incomplete() -> None:
    plan = build_next_probe({"run_ref": "run_bad"})

    assert plan.diagnosis["primary"] == "artifact_incomplete"
    assert "status" in plan.missing_fields
    assert "command" not in plan.recommendation


def test_sweep_and_next_probe_cli_write_outputs(tmp_path: Path, monkeypatch) -> None:
    init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    ProjectManager.init(tmp_path, name="self-ground")
    hypothesis = tmp_path / "hypothesis.json"
    hypothesis.write_text(
        json.dumps(
            {
                "wb_ref": "hyp_cli",
                "title": "cli hypothesis",
                "units": ["unit_1"],
                "example_bundle_ref": "bundle_1",
                "control_bundle_ref": "ctrl_1",
                "expected_effect": "target_delta > controls",
                "required_controls": ["negation_removed"],
            }
        )
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_ref": "run_cli",
                "status": "insufficient_evidence",
                "tried_axes": {"layers": ["1", "2"], "patch_modes": ["delta"]},
                "available_axes": {"layers": ["0", "1", "2"], "patch_modes": ["delta", "direct"]},
                "backend_capabilities": {"direct_patch": True},
            }
        )
    )
    (run_dir / "control_metrics.json").write_text(
        json.dumps({"target_delta": 0.5, "matched_control_delta": 0.45, "family_min_gap": -0.02})
    )
    runner = CliRunner()

    sweep = runner.invoke(
        app,
        [
            "sweep",
            str(hypothesis),
            "--axis",
            "layer=0,1",
            "--axis",
            "patch_mode=direct",
            "--dry-run",
        ],
    )
    next_probe = runner.invoke(app, ["next-probe", str(run_dir)])

    assert sweep.exit_code == 0, sweep.output
    assert next_probe.exit_code == 0, next_probe.output
    assert (run_dir / "next_probe.yaml").exists()
    assert (run_dir / "next_probe.md").exists()

