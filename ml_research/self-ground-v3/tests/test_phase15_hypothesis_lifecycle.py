import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from mwb.cli import app
from mwb.project import Project, ProjectManager


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def write_hypothesis(project: Project, hypothesis_ref: str = "hyp_phase15") -> Path:
    path = project.mechanism_dir / "hypotheses" / f"{hypothesis_ref}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "wb_ref": hypothesis_ref,
                "wb_type": "Hypothesis",
                "wb_version": "1",
                "created_at": "2026-06-26T00:00:00Z",
                "parents": ["unit_phase15", "bundle_targets", "bundle_controls"],
                "metadata": {},
                "title": "Phase 15 lifecycle hypothesis",
                "units": ["unit_phase15"],
                "example_bundle_ref": "bundle_targets",
                "control_bundle_ref": "bundle_controls",
                "expected_effect": "target_delta > matched_control_delta",
                "required_controls": ["matched_controls"],
                "alternative_explanations": ["control_leaky"],
                "requested_evidence_tier": "causal_necessity",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def write_run_with_blockers(project: Project, hypothesis_ref: str = "hyp_phase15") -> Path:
    run_ref = "run_phase15_control_leaky"
    run_dir = project.mechanism_dir / "runs" / run_ref
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_ref": run_ref,
                "source_hypothesis_ref": hypothesis_ref,
                "status": "insufficient_evidence",
                "evidence_posture": "diagnostic_insufficient",
                "claim_bearing": False,
                "created_at": "2026-06-26T00:00:00Z",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "control_metrics.json").write_text(
        json.dumps(
            {
                "target_delta": 0.62,
                "matched_control_delta": 0.59,
                "specificity_gap": 0.03,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "blocker_report.json").write_text(
        json.dumps(
            {
                "wb_ref": "blocker_phase15",
                "run_ref": run_ref,
                "blockers": ["control_leaky"],
                "primary_blocker": "control_leaky",
                "blocking_metrics": [
                    {
                        "name": "matched_control_delta",
                        "value": 0.59,
                        "threshold": "< 0.8 * target_delta",
                        "status": "fail",
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_hypothesis_state_separates_workflow_evidence_and_claim_status() -> None:
    from mwb.domain.objects import HypothesisState

    state = HypothesisState(
        wb_ref="hypstate_phase15",
        hypothesis_ref="hyp_phase15",
        state="triaged",
        evidence_tier="association",
        claim_status="single_run_evidence",
    )

    assert state.state == "triaged"
    assert state.evidence_tier == "association"
    assert state.claim_status == "single_run_evidence"


def test_hypothesis_transition_cli_validates_order_and_claimable_approval(
    tmp_path: Path, monkeypatch
) -> None:
    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    write_hypothesis(project)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    triage = runner.invoke(
        app,
        [
            "hypothesis",
            "transition",
            "hyp_phase15",
            "--to-state",
            "triaged",
            "--evidence-tier",
            "association",
        ],
    )
    invalid = runner.invoke(
        app,
        ["hypothesis", "transition", "hyp_phase15", "--to-state", "generalized"],
    )

    assert triage.exit_code == 0, triage.output
    assert invalid.exit_code == 1

    state_path = project.mechanism_dir / "hypotheses" / "hyp_phase15_lifecycle.json"
    state_payload = json.loads(state_path.read_text())
    state_payload["state"] = "generalized"
    state_path.write_text(json.dumps(state_payload, sort_keys=True) + "\n")

    unapproved = runner.invoke(
        app,
        ["hypothesis", "transition", "hyp_phase15", "--to-state", "claimable"],
    )
    approved = runner.invoke(
        app,
        [
            "hypothesis",
            "transition",
            "hyp_phase15",
            "--to-state",
            "claimable",
            "--approved-by",
            "local_user",
            "--decision-ref",
            "D001",
        ],
    )

    assert unapproved.exit_code == 1
    assert approved.exit_code == 0, approved.output
    receipt = json.loads(approved.output)["receipt"]
    assert receipt["from_state"] == "generalized"
    assert receipt["to_state"] == "claimable"
    assert receipt["approved_by"] == "local_user"


def test_hypothesis_explain_writes_live_alternatives_from_blockers(
    tmp_path: Path, monkeypatch
) -> None:
    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    write_hypothesis(project)
    run_dir = write_run_with_blockers(project)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, ["hypothesis", "explain", run_dir.name])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    alternatives = payload["alternatives"]
    assert alternatives[0]["explanation_id"] == "control_leaky"
    assert "matched_control_delta" in alternatives[0]["evidence_for"][0]
    assert alternatives[0]["next_test"]
    run_alternatives = run_dir / "alternative_explanations.json"
    hyp_alternatives = project.mechanism_dir / "hypotheses" / "hyp_phase15_alternatives.json"
    assert run_alternatives.exists()
    assert hyp_alternatives.exists()


def test_hypothesis_lifecycle_rebuild_restores_state_and_alternatives(
    tmp_path: Path, monkeypatch
) -> None:
    from mwb.sqlite_index import fetch_payload, rebuild_sqlite_index

    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    write_hypothesis(project)
    run_dir = write_run_with_blockers(project)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    runner.invoke(app, ["hypothesis", "transition", "hyp_phase15", "--to-state", "triaged"])
    runner.invoke(app, ["hypothesis", "explain", run_dir.name])
    rebuilt = tmp_path / "rebuilt.sqlite"
    report = rebuild_sqlite_index(project, output_path=rebuilt)

    assert report["counts"]["hypothesis_states"] == 1
    assert report["counts"]["alternative_explanations"] == 1
    assert fetch_payload(rebuilt, "hypothesis_states", "hyp_phase15")["state"] == "triaged"
    alternatives = fetch_payload(rebuilt, "alternative_explanations", "hyp_phase15")
    assert alternatives["alternatives"][0]["explanation_id"] == "control_leaky"
