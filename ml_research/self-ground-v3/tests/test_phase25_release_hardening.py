import json
import re
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from mwb.cli import app
from mwb.project import ProjectManager
from mwb.reference_benchmarks import ReferenceBenchmarkService
from mwb.sqlite_index import fetch_payload, rebuild_sqlite_index
from mwb.workflows.draft_guard import check_draft_text

ROOT = Path(__file__).resolve().parents[1]


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def test_release_regression_suite_blocks_known_false_positives() -> None:
    benchmark = ReferenceBenchmarkService().run_suite()
    negative_control = next(
        task for task in benchmark.tasks if task["task_id"] == "negative_control_surface_confound"
    )
    card = json.loads(
        (ROOT / "docs" / "fixtures" / "runs" / "control_leaky" / "mechanism_card.json").read_text(
            encoding="utf-8"
        )
    )
    draft = "Feature F implements negation. [CLAIM:claim_run_fixture_control_leaky]"
    draft_report = check_draft_text(draft, {"claim_run_fixture_control_leaky": card})

    assert negative_control["classification"] == "false_positive_blocked"
    assert "tempting_confound" in negative_control["blockers"]
    assert draft_report["status"] == "blocked"
    assert "implements" in draft_report["findings"][0]["blocked_terms"]


def test_release_legacy_mechanism_state_without_new_refs_rebuilds(tmp_path: Path) -> None:
    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    adapter_dir = tmp_path / ".mechanism" / "adapters" / "legacy_adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "manifest.json").write_text(
        json.dumps(
            {
                "adapter_name": "LegacyAdapter",
                "adapter_version": "0.0.1",
                "claim_bearing": {"supported": False, "required_conformance": []},
                "capabilities": {"legacy_read": True},
                "limitations": ["legacy record without stable manifest_ref"],
                "package": {"name": "legacy", "version": None},
            }
        ),
        encoding="utf-8",
    )
    (adapter_dir / "backend_versions.json").write_text(
        json.dumps(
            {
                "adapter_name": "LegacyAdapter",
                "adapter_version": "0.0.1",
                "package_versions": {},
                "python_version": "3.11.0",
                "platform": "legacy",
                "cuda_available": False,
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )

    report = rebuild_sqlite_index(project, output_path=tmp_path / "rebuilt.sqlite")

    assert report["counts"]["adapter_manifests"] == 1
    assert report["counts"]["backend_versions"] == 1
    assert (
        fetch_payload(tmp_path / "rebuilt.sqlite", "adapter_manifests", "legacy_adapter")[
            "adapter_name"
        ]
        == "LegacyAdapter"
    )
    assert (
        fetch_payload(tmp_path / "rebuilt.sqlite", "backend_versions", "legacy_adapter")[
            "device"
        ]
        == "cpu"
    )


def test_release_docs_links_and_report_are_current() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    usage = (ROOT / "docs" / "USAGE.md").read_text(encoding="utf-8")
    report = ROOT / "docs" / "RELEASE_HARDENING_REPORT.md"

    assert "docs/RELEASE_HARDENING_REPORT.md" in readme
    assert report.exists()
    for doc_path in sorted(set(re.findall(r"docs/[A-Za-z0-9_./-]+\\.md", readme))):
        assert (ROOT / doc_path).exists(), doc_path
    for command in [
        "uv run mwb benchmark framework",
        "uv run mwb policy check",
        "uv run mwb adapter conformance nnsight",
        "uv run mwb adapter conformance pyvene",
        "uv run mwb adapter conformance neuronpedia",
    ]:
        assert command in usage

    report_text = report.read_text(encoding="utf-8")
    assert "119 passed, 3 skipped" in report_text
    assert "adapter_manifests: 5" in report_text
    assert "backend_versions: 5" in report_text


def test_public_cli_help_snapshot_contains_release_commands() -> None:
    runner = CliRunner()
    root_help = runner.invoke(app, ["--help"])
    adapter_help = runner.invoke(app, ["adapter", "conformance", "--help"])
    pyvene_help = runner.invoke(app, ["adapter", "conformance", "pyvene", "--help"])

    assert root_help.exit_code == 0, root_help.output
    assert adapter_help.exit_code == 0, adapter_help.output
    assert pyvene_help.exit_code == 0, pyvene_help.output
    for command in [
        "benchmark",
        "policy",
        "diagnose",
        "next-probe",
        "run-probe",
        "draft-check",
        "repair-index",
    ]:
        assert command in root_help.output
    for command in ["transformer-lens", "saelens", "nnsight", "pyvene", "neuronpedia"]:
        assert command in adapter_help.output
    for option in ["--module-path", "--intervention-kind", "--dry-run"]:
        assert option in pyvene_help.output
