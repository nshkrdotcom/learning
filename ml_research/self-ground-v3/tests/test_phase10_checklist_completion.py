import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from mwb.cli import app
from mwb.project import ProjectManager
from mwb.sqlite_index import fetch_payload, rebuild_sqlite_index
from mwb.workflows.cards import card_from_run, language_for_tier, write_card
from mwb.workflows.draft_guard import check_draft_text
from mwb.workflows.self_ground_ingest import validate_self_ground_artifacts


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_hypothesis(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "wb_ref": "hyp_completion",
                "title": "completion hypothesis",
                "units": ["unit_1"],
                "example_bundle_ref": "bundle_1",
                "control_bundle_ref": "ctrl_1",
                "expected_effect": "target_delta > controls",
                "required_controls": ["negation_removed"],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def write_run(path: Path, *, status: str = "insufficient_evidence") -> None:
    path.mkdir()
    (path / "run_manifest.json").write_text(
        json.dumps({"run_ref": "run_completion", "status": status}) + "\n",
        encoding="utf-8",
    )
    (path / "control_metrics.json").write_text(
        json.dumps(
            {
                "target_delta": 0.5,
                "matched_control_delta": 0.45,
                "specificity_gap": 0.05,
                "family_min_gap": -0.02,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def write_full_e004_fixture(path: Path) -> None:
    comparison = path / "comparison"
    forensics = path / "forensics"
    comparison.mkdir(parents=True)
    forensics.mkdir()
    row = {
        "layer": "blocks.2.hook_resid_post",
        "sae_release": "pythia-70m-deduped-res-sm",
        "sae_id": "blocks.2.hook_resid_post",
        "feature_selection_mode": "ensemble-specificity",
        "operation": "ablate,amplify",
        "control_suite": "multi_control",
        "run_status": "completed",
        "claim_status": "insufficient_evidence",
        "valid_tasks": 69,
        "behavioral_rows": 5520,
        "skipped_rows": 0,
        "baseline_pass_rate": 1.0,
        "top_target_delta": 0.9,
        "top_control_delta": 0.82,
        "specificity_gap": 0.08,
        "top_vs_control_ratio": 1.09,
        "density_control_gap": 0.0,
        "multi_control_min_gap": -0.01,
        "family_min_specificity_gap": -0.04,
        "passes_all_controls": False,
        "limitations": "Multi-control evidence failed.",
        "artifact_paths": "runs/e004_specificity_rescue_matrix/eval/block2",
    }
    (path / "capability.json").write_text(json.dumps({"cuda_available": False}) + "\n")
    (path / "matrix_run_summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "attempted_cells": 1,
                "completed_cells": 1,
                "blocked_cells": 0,
                "layers": ["blocks.2.hook_resid_post"],
                "feature_selection_modes": ["ensemble-specificity"],
                "operations": ["ablate,amplify"],
                "control_suite": "multi_control",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (comparison / "comparison.json").write_text(
        json.dumps(
            {
                "interpretation": "current_sae_model_layer_search_insufficient",
                "best_run": row,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (comparison / "matrix_summary.json").write_text(json.dumps([row]) + "\n", encoding="utf-8")
    summary_header = ",".join(row.keys())
    summary_line = ",".join(str(value) for value in row.values())
    for name in [
        "matrix_summary.csv",
        "best_runs_by_family.csv",
        "best_runs_by_specificity.csv",
        "blocked_runs.csv",
    ]:
        (comparison / name).write_text(f"{summary_header}\n{summary_line}\n", encoding="utf-8")
    (comparison / "claim_adjudication.md").write_text("# E004 Claim Adjudication\n")
    for name, header, line in [
        (
            "control_suite_breakdown.csv",
            "control_suite,n_rows,n_tasks,target_delta_mean,control_delta_mean,specificity_gap_mean,control_dominant_rows",
            "multi_control,10,5,0.9,0.82,0.08,3",
        ),
        (
            "family_breakdown.csv",
            "family,n_rows,n_tasks,target_delta_mean,control_delta_mean,specificity_gap_mean,control_dominant_rows",
            "negation_removed,10,5,0.9,0.82,0.08,3",
        ),
        (
            "feature_breakdown.csv",
            "feature_set,n_rows,n_tasks,target_delta_mean,control_delta_mean,specificity_gap_mean,control_dominant_rows",
            "top,10,5,0.9,0.82,0.08,3",
        ),
        (
            "task_outlier_table.csv",
            "run_name,task_id,family,template_id,token_pair,feature_set_label,control_suite,target_absolute_delta,control_absolute_delta,specificity_gap",
            "run,task,negation_removed,tpl,no/not,top,multi_control,0.9,0.82,0.08",
        ),
        (
            "template_breakdown.csv",
            "template,n_rows,n_tasks,target_delta_mean,control_delta_mean,specificity_gap_mean,control_dominant_rows",
            "tpl,10,5,0.9,0.82,0.08,3",
        ),
        (
            "token_pair_breakdown.csv",
            "token_pair,n_rows,n_tasks,target_delta_mean,control_delta_mean,specificity_gap_mean,control_dominant_rows",
            "no/not,10,5,0.9,0.82,0.08,3",
        ),
    ]:
        (forensics / name).write_text(f"{header}\n{line}\n", encoding="utf-8")
    (forensics / "forensics_summary.md").write_text("# Forensics\n", encoding="utf-8")


def test_ipython_resume_cli_links_sessions_and_continues_capture(
    tmp_path: Path, monkeypatch
) -> None:
    init_git_repo(tmp_path)
    ProjectManager.init(tmp_path, name="self-ground")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    first = runner.invoke(app, ["ipython", "--execute", "obj = ctx.note('first')"])
    first_session = sorted((tmp_path / ".mechanism" / "sessions").glob("sess_*"))[-1]

    second = runner.invoke(
        app,
        ["ipython", "--resume", first_session.name, "--execute", "obj = ctx.note('second')"],
    )
    session_dirs = sorted((tmp_path / ".mechanism" / "sessions").glob("sess_*"))
    second_session = next(
        path
        for path in session_dirs
        if json.loads((path / "session.json").read_text()).get("resumed_from_session_ref")
        == first_session.name
    )
    session_payload = json.loads((second_session / "session.json").read_text())
    object_events = jsonl(second_session / "namespace_objects.jsonl")

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert second_session != first_session
    assert session_payload["resumed_from_session_ref"] == first_session.name
    assert object_events[0]["object_type"] == "Note"


def test_ctx_record_and_note_are_captured_from_ipython(tmp_path: Path, monkeypatch) -> None:
    init_git_repo(tmp_path)
    ProjectManager.init(tmp_path, name="self-ground")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ipython",
            "--execute",
            "note = ctx.note('important')",
            "--execute",
            "recorded = ctx.record(note, name='reviewed-note')",
        ],
    )
    session = sorted((tmp_path / ".mechanism" / "sessions").glob("sess_*"))[-1]
    events = jsonl(session / "namespace_objects.jsonl")

    assert result.exit_code == 0, result.output
    assert [event["variable_name"] for event in events] == ["note", "recorded"]
    assert events[-1]["object_type"] == "Note"


def test_sweep_dry_run_writes_full_non_claim_bearing_artifact_set(
    tmp_path: Path, monkeypatch
) -> None:
    init_git_repo(tmp_path)
    ProjectManager.init(tmp_path, name="self-ground")
    monkeypatch.chdir(tmp_path)
    hypothesis = tmp_path / "hypothesis.json"
    write_hypothesis(hypothesis)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "sweep",
            str(hypothesis),
            "--axis",
            "layer=0,1",
            "--axis",
            "feature_selection_mode=top-absolute",
            "--axis",
            "operation=ablate",
            "--axis",
            "patch_mode=direct",
            "--axis",
            "amplification_factor=1.0",
            "--axis",
            "control_family=negation_removed",
            "--dry-run",
        ],
    )
    payload = json.loads(result.output)
    run_dir = Path(payload["run_dir"])

    assert result.exit_code == 0, result.output
    assert payload["claim_bearing"] is False
    for name in [
        "sweep_config.json",
        "verification_results.jsonl",
        "control_metrics.json",
        "intervention_receipts.jsonl",
        "run_manifest.json",
        "blocker_report.json",
    ]:
        assert (run_dir / name).exists(), name
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "dry_run"
    assert manifest["claim_bearing"] is False
    assert len(jsonl(run_dir / "verification_results.jsonl")) == 2
    assert len(jsonl(run_dir / "intervention_receipts.jsonl")) == 2


def test_language_for_all_evidence_tiers_is_explicit() -> None:
    for tier in [
        "association",
        "projection",
        "causal_necessity",
        "causal_sufficiency",
        "mediation",
        "generalization",
        "mechanism",
    ]:
        allowed, blocked = language_for_tier(tier, [])
        assert allowed, tier
        assert blocked or tier == "mechanism", tier
    assert "implements" in language_for_tier("mechanism", [])[0]
    assert "implements" in language_for_tier("causal_sufficiency", [])[1]


def test_card_writes_scientific_debt_records(tmp_path: Path) -> None:
    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    run_dir = tmp_path / "run"
    write_run(run_dir)

    card = card_from_run(run_dir)
    write_card(run_dir, card, mechanism_dir=project.mechanism_dir)
    debt = json.loads((run_dir / "scientific_debt.json").read_text())

    assert debt["run_ref"] == "run_completion"
    assert "control_leaky" in debt["items"][0]["blocker"]


def test_draft_guard_supports_caveated_unknown_and_missing_statuses() -> None:
    report = check_draft_text(
        "Feature F is associated with target behavior. [CLAIM:claim_1]",
        {
            "claim_1": {
                "blocked_language": ["implements"],
                "allowed_language": ["is associated with"],
                "required_caveats": ["control leakage remains unresolved"],
            }
        },
    )
    unknown = check_draft_text("Feature F helps. [CLAIM:claim_missing]", {})
    no_tags = check_draft_text("Feature F helps.", {})

    assert report["status"] == "caveated"
    assert report["findings"][0]["required_caveats"] == ["control leakage remains unresolved"]
    assert unknown["status"] == "missing_card"
    assert no_tags["status"] == "unknown_claim"


def test_self_ground_validation_requires_comparison_and_forensics_csvs(tmp_path: Path) -> None:
    source = tmp_path / "e004_specificity_rescue_matrix"
    write_full_e004_fixture(source)

    statuses = validate_self_ground_artifacts(source)

    assert statuses["comparison/best_runs_by_family.csv"]["status"] == "present"
    assert statuses["comparison/best_runs_by_specificity.csv"]["row_count"] == 1
    assert statuses["forensics/control_suite_breakdown.csv"]["status"] == "present"
    assert statuses["forensics/task_outlier_table.csv"]["row_count"] == 1


def test_sqlite_rebuild_restores_indexed_files(tmp_path: Path, monkeypatch) -> None:
    init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    runner = CliRunner()
    result = runner.invoke(app, ["ipython", "--execute", "obj = ctx.note('rebuild')"])
    rebuilt = tmp_path / "rebuilt.sqlite"

    report = rebuild_sqlite_index(project, output_path=rebuilt)

    assert result.exit_code == 0, result.output
    assert report["status"] == "ok"
    assert report["counts"]["sessions"] == 1
    assert report["counts"]["cells"] == 1
    assert report["counts"]["objects"] == 1
    session_dir = sorted((tmp_path / ".mechanism" / "sessions").glob("sess_*"))[-1]
    object_ref = jsonl(session_dir / "namespace_objects.jsonl")[0]["object_ref"]
    assert fetch_payload(rebuilt, "objects", object_ref)["wb_type"] == "Note"

    cli_rebuilt = tmp_path / "cli-rebuilt.sqlite"
    cli = runner.invoke(app, ["rebuild-index", "--output", str(cli_rebuilt)])
    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.output)["status"] == "ok"
    assert cli_rebuilt.exists()
