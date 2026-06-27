import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from mwb.cli import app
from mwb.project import ProjectManager


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def bundle_payload() -> dict:
    return {
        "name": "test_negation",
        "domain": "negation",
        "examples": [
            {
                "id": "tgt_1",
                "prompt": "The answer is not yes.",
                "target": "no",
                "foil": "yes",
                "family": "target_negation",
                "baseline_margin": 0.4,
            },
            {
                "id": "tgt_2",
                "prompt": "The statement is not true.",
                "target": "false",
                "foil": "true",
                "family": "target_negation",
                "baseline_margin": 0.3,
            },
        ],
        "controls": {
            "negation_removed": [
                {
                    "id": "ctrl_1",
                    "prompt": "The answer is yes.",
                    "target": "yes",
                    "family": "negation_removed",
                    "baseline_margin": 0.35,
                }
            ],
            "paraphrase_negation": [
                {
                    "id": "ctrl_2",
                    "prompt": "It is false that the answer is yes.",
                    "target": "no",
                    "family": "paraphrase_negation",
                    "baseline_margin": 0.2,
                }
            ],
        },
    }


def test_token_validity_audit_fails_missing_target() -> None:
    from mwb.bundle_audit import BundleAuditService

    payload = bundle_payload()
    payload["examples"][0]["target"] = ""

    report = BundleAuditService().audit_payload(payload)

    assert report.status == "fail"
    assert "token_validation_failed" in report.blockers
    token_check = next(check for check in report.checks if check["name"] == "token_validity")
    assert token_check["invalid_count"] == 1


def test_role_balance_warns_and_proposes_missing_controls() -> None:
    from mwb.bundle_audit import BundleAuditService

    payload = bundle_payload()
    payload["controls"]["negation_removed"] = []

    report = BundleAuditService().audit_payload(payload)

    assert report.status == "warn"
    assert "role_balance_low" in report.warnings
    assert any(proposal["kind"] == "add_control_examples" for proposal in report.proposals)


def test_contaminated_control_detection_flags_negation_removed_leak() -> None:
    from mwb.bundle_audit import BundleAuditService

    payload = bundle_payload()
    payload["controls"]["negation_removed"][0]["prompt"] = "The answer is not yes."

    report = BundleAuditService().audit_payload(payload)
    contamination = report.contamination_report

    assert contamination["status"] == "fail"
    assert contamination["contaminated_count"] == 1
    assert "control_contamination" in report.blockers


def test_baseline_margin_check_blocks_low_margins() -> None:
    from mwb.bundle_audit import BundleAuditService

    payload = bundle_payload()
    payload["examples"][0]["baseline_margin"] = 0.01

    report = BundleAuditService().audit_payload(payload)

    assert report.status == "fail"
    assert "baseline_margin_low" in report.blockers


def test_rebalance_dry_run_generates_heldout_and_control_proposals() -> None:
    from mwb.bundle_audit import BundleAuditService

    proposal = BundleAuditService().rebalance_payload(bundle_payload(), dry_run=True)

    assert proposal.dry_run is True
    assert any(item["kind"] == "heldout_template" for item in proposal.proposals)
    assert any(item["kind"] == "add_control_examples" for item in proposal.proposals)


def test_bundle_audit_cli_writes_report_and_links_e004_forensics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    init_git_repo(tmp_path)
    ProjectManager.init(tmp_path, name="self-ground")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, ["bundle", "audit", "negation_phase3_calibrated"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["bundle_name"] == "phase3_calibrated"
    assert payload["source_links"]["self_ground_e004_forensics"].endswith("forensics_summary.md")
    assert (tmp_path / ".mechanism" / "bundle_audits" / "latest_bundle_audit.json").exists()
