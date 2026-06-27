import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from mwb.cli import app
from mwb.project import ProjectManager
from mwb.reference_benchmarks import ReferenceBenchmarkService
from mwb.sqlite_index import fetch_payload, rebuild_sqlite_index


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def task_by_id(report: object, task_id: str) -> dict:
    tasks = report.tasks
    return next(task for task in tasks if task["task_id"] == task_id)


def test_toy_known_mechanism_classification_recovers_planted_unit(tmp_path: Path) -> None:
    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")

    report = ReferenceBenchmarkService(project).run_suite("toy")

    task = task_by_id(report, "toy_residual_sign")
    assert report.status == "pass"
    assert task["classification"] == "mechanism_found"
    assert task["passed"] is True
    assert task["found_units"] == ["unit_direct_writer"]
    assert task["ground_truth"]["mechanism_units"] == ["unit_direct_writer"]
    assert task["scores"]["top_exact_unit"] == "unit_direct_writer"


def test_tempting_false_positive_confound_is_blocked(tmp_path: Path) -> None:
    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")

    report = ReferenceBenchmarkService(project).run_suite("toy")

    task = task_by_id(report, "negative_control_surface_confound")
    assert task["classification"] == "false_positive_blocked"
    assert task["passed"] is True
    assert "tempting_confound" in task["blockers"]
    assert task["scores"]["top_proxy_unit"] == "unit_surface_token"
    assert task["scores"]["top_exact_unit"] != "unit_surface_token"


def test_synthetic_sae_split_and_absorption_are_detected(tmp_path: Path) -> None:
    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")

    report = ReferenceBenchmarkService(project).run_suite("toy")

    task = task_by_id(report, "synthetic_sae_split_absorption")
    check_names = {check["name"]: check for check in task["checks"]}
    assert task["classification"] == "dictionary_artifact_detected"
    assert task["passed"] is True
    assert check_names["feature_split"]["status"] == "pass"
    assert check_names["feature_absorption"]["status"] == "pass"
    assert check_names["feature_split"]["metrics"]["split_latents"] == ["negation"]
    assert check_names["feature_absorption"]["metrics"]["absorbed_features"] == [
        "feature_absorbed_negation_sentiment"
    ]


def test_framework_benchmark_cli_writes_report_and_sqlite_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    runner = CliRunner()

    result = runner.invoke(app, ["benchmark", "framework"])

    assert result.exit_code == 0, result.output
    output = json.loads(result.output)
    assert output["wb_type"] == "ReferenceBenchmarkReport"
    assert output["suite"] == "toy"
    assert output["status"] == "pass"
    assert output["summary"]["task_count"] == 3
    assert output["calibration"]["null_seed_count"] > 0
    assert "proxy_vs_exact_correlation" in output["calibration"]
    report_path = project.mechanism_dir / "benchmarks" / "latest_framework_benchmark.json"
    assert report_path.exists()

    restored = rebuild_sqlite_index(project, output_path=tmp_path / "rebuilt.sqlite")

    assert restored["counts"]["benchmark_reports"] == 1
    assert restored["counts"]["reference_tasks"] == 3
    indexed = fetch_payload(tmp_path / "rebuilt.sqlite", "benchmark_reports", output["wb_ref"])
    assert indexed["calibration"]["fdr_adjusted_p_value"] <= 0.1
