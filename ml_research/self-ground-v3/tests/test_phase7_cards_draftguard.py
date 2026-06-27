import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from mwb.cli import app
from mwb.project import ProjectManager
from mwb.workflows.cards import card_from_run
from mwb.workflows.draft_guard import check_draft_text


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# repo\n")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def write_control_leaky_run(path: Path) -> None:
    path.mkdir()
    (path / "run_manifest.json").write_text(
        json.dumps({"run_ref": "run_fixture_control_leaky", "status": "insufficient_evidence"})
    )
    (path / "control_metrics.json").write_text(
        json.dumps({"target_delta": 0.5, "matched_control_delta": 0.45, "family_min_gap": -0.02})
    )
    (path / "next_probe.yaml").write_text("diagnosis:\n  primary: control_leaky\n")


def test_card_from_control_leaky_run_blocks_mechanism_language(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_control_leaky_run(run_dir)

    card = card_from_run(run_dir)

    assert card.status == "insufficient_evidence"
    assert card.evidence_tier == "association"
    assert "is associated with" in card.allowed_language
    assert "implements" in card.blocked_language
    assert card.metadata["claim_ref"] == "claim_run_fixture_control_leaky"


def test_draft_guard_blocks_overclaiming() -> None:
    card = {
        "claim_ref": "claim_1",
        "evidence_tier": "association",
        "blocked_language": ["implements"],
        "allowed_language": ["is associated with"],
        "status": "insufficient_evidence",
    }
    report = check_draft_text("Feature F implements negation. [CLAIM:claim_1]", {"claim_1": card})

    assert report["status"] == "blocked"
    assert report["findings"][0]["claim_ref"] == "claim_1"
    assert "implements" in report["findings"][0]["blocked_terms"]


def test_card_and_draft_check_cli(tmp_path: Path, monkeypatch) -> None:
    init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    ProjectManager.init(tmp_path, name="self-ground")
    run_dir = tmp_path / "run"
    write_control_leaky_run(run_dir)
    draft = tmp_path / "draft.md"
    draft.write_text("Feature F implements negation. [CLAIM:claim_run_fixture_control_leaky]\n")
    runner = CliRunner()

    card = runner.invoke(app, ["card", str(run_dir)])
    draft_check = runner.invoke(app, ["draft-check", str(draft)])

    assert card.exit_code == 0, card.output
    assert draft_check.exit_code == 1, draft_check.output
    assert (run_dir / "mechanism_card.json").exists()
    assert json.loads(draft_check.output)["status"] == "blocked"

