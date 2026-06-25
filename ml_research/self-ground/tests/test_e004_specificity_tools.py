from __future__ import annotations

import csv
import importlib.util
import json
from argparse import Namespace
from pathlib import Path


def _load_script(name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare_matrix = _load_script("compare_e004_matrix").compare_matrix
diagnose = _load_script("diagnose_e003_specificity_failure").diagnose
run_e004_module = _load_script("run_e004_specificity_rescue_matrix")
run_matrix = run_e004_module.run_matrix
write_forensics = _load_script("write_specificity_forensics_report").write_forensics


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_run(
    path: Path,
    *,
    claim_status: str = "insufficient_evidence",
    gap: float = -0.2,
) -> None:
    _write_json(
        path / "config.json",
        {
            "hook_point": "blocks.2.hook_resid_post",
            "sae_release": "pythia-70m-deduped-res-sm",
            "sae_id": "blocks.2.hook_resid_post",
            "feature_selection_mode": "top-absolute",
            "operations": ["ablate"],
            "control_suite": "multi_control",
        },
    )
    _write_json(path / "mechanism_report.json", {"claim_status": claim_status, "limitations": []})
    _write_json(
        path / "behavioral_task_validation.json",
        {"summary": {"valid_tasks": 3, "valid_by_family": {"sentiment_negation": 1}}},
    )
    _write_json(path / "skipped_behavioral_rows.json", {"n_skipped_rows": 0})
    _write_jsonl(
        path / "behavioral_tasks.jsonl",
        [
            {
                "id": "task_0",
                "family": "sentiment_negation",
                "target_tokens": [" bad"],
                "foil_tokens": [" good"],
                "metadata": {"template_id": "sentiment_template"},
            }
        ],
    )
    target = 0.3
    control = target - gap
    _write_jsonl(
        path / "baseline_task_scores.jsonl",
        [
            {
                "task_id": "task_0",
                "family": "sentiment_negation",
                "baseline_prompt_contrast": 1.0,
            }
        ],
    )
    _write_jsonl(
        path / "behavioral_intervention_results.jsonl",
        [
            {
                "task_id": "task_0",
                "family": "sentiment_negation",
                "feature_set_label": "top",
                "control_suite": "matched_non_negation_current",
                "target_tokens": [" bad"],
                "foil_tokens": [" good"],
                "target_absolute_delta": target,
                "control_absolute_delta": control,
            }
        ],
    )
    summary_fields = [
        "feature_set_label",
        "feature_selection_method",
        "operation",
        "factor",
        "patch_mode",
        "control_suite",
        "family",
        "n_tasks",
        "target_absolute_delta_mean",
        "control_absolute_delta_mean",
        "specificity_gap_mean",
        "collateral_ratio_mean",
    ]
    rows = [
        {
            "feature_set_label": "top",
            "feature_selection_method": "ranking_abs_score_top_k",
            "operation": "ablate",
            "factor": "",
            "patch_mode": "delta",
            "control_suite": "matched_non_negation_current",
            "family": "__all__",
            "n_tasks": 1,
            "target_absolute_delta_mean": target,
            "control_absolute_delta_mean": control,
            "specificity_gap_mean": gap,
            "collateral_ratio_mean": control / target,
        },
        {
            "feature_set_label": "top",
            "feature_selection_method": "ranking_abs_score_top_k",
            "operation": "ablate",
            "factor": "",
            "patch_mode": "delta",
            "control_suite": "matched_non_negation_current",
            "family": "sentiment_negation",
            "n_tasks": 1,
            "target_absolute_delta_mean": target,
            "control_absolute_delta_mean": control,
            "specificity_gap_mean": gap,
            "collateral_ratio_mean": control / target,
        },
    ]
    _write_csv(path / "behavioral_summary.csv", rows, summary_fields)


def test_e003_specificity_diagnosis_detects_global_control_dominance(tmp_path) -> None:
    run_dir = tmp_path / "run"
    ranking_dir = tmp_path / "ranking"
    calibration_dir = tmp_path / "calibration"
    _synthetic_run(run_dir, gap=-0.2)
    _write_json(ranking_dir / "feature_sets.json", {})
    _write_csv(
        ranking_dir / "feature_rankings.csv",
        [{"feature_id": "sae_0", "score": 1.0, "abs_score": 1.0}],
        ["feature_id", "score", "abs_score"],
    )
    _write_json(calibration_dir / "calibration_summary.json", {"kept_total": 3})

    summary = diagnose(run_dir, ranking_dir, calibration_dir, tmp_path / "out")

    assert "control_dominates_globally" in summary["diagnosis_labels"]
    assert (tmp_path / "out" / "specificity_summary.json").exists()
    assert (tmp_path / "out" / "diagnosis.md").exists()


def test_compare_e004_matrix_handles_completed_and_blocked_cells(tmp_path) -> None:
    e003 = tmp_path / "e003"
    matrix = tmp_path / "matrix"
    completed = matrix / "eval" / "block2_top"
    blocked = matrix / "eval" / "block3_blocked"
    _synthetic_run(e003, gap=-0.1)
    _synthetic_run(completed, claim_status="insufficient_evidence", gap=0.1)
    _write_json(blocked / "CELL_BLOCKED.json", {"reason": "ranking failed"})

    payload = compare_matrix(matrix, e003, tmp_path / "comparison")

    assert payload["attempted_cells"] == 2
    assert payload["completed_cells"] == 1
    assert payload["blocked_cells"] == 1
    assert (tmp_path / "comparison" / "claim_adjudication.md").exists()


def test_forensics_report_writes_breakdowns(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _synthetic_run(run_dir, gap=-0.2)

    summary = write_forensics([run_dir], tmp_path / "forensics")

    assert summary["n_rows"] == 1
    assert (tmp_path / "forensics" / "forensics_summary.md").exists()
    assert (tmp_path / "forensics" / "control_suite_breakdown.csv").exists()


def test_e004_orchestrator_blocks_when_cuda_unavailable(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        run_e004_module,
        "_cuda_available",
        lambda: (False, {"cuda_available": False}),
    )
    args = Namespace(
        device="cuda",
        task_file=tmp_path / "tasks.jsonl",
        task_bank_calibration_dir=tmp_path / "calibration",
        layers="blocks.2.hook_resid_post",
        feature_selection_modes="top-absolute",
        operations="ablate",
        amplify_factors="2.0",
        control_suite="multi_control",
        ranking_top_k=10,
        eval_top_k=2,
        min_calibrated_per_family=1,
        min_family_consistency=2,
        random_seeds="7,11,13",
        out_root=tmp_path / "matrix",
        force=False,
    )

    payload = run_matrix(args)

    assert payload["status"] == "blocked"
    assert (tmp_path / "matrix" / "BLOCKED.json").exists()


def test_e004_orchestrator_resumes_completed_eval_cells(tmp_path, monkeypatch) -> None:
    completed = tmp_path / "matrix" / "eval" / "block2_absolute_ablate_multi"
    _write_json(completed / "mechanism_report.json", {"claim_status": "insufficient_evidence"})
    _write_json(completed / "inspection_summary.json", {"claim_status": "insufficient_evidence"})

    calls: list[list[str]] = []

    def fail_if_called(command: list[str]):
        calls.append(command)
        raise AssertionError("completed E004 cell should not be recomputed")

    monkeypatch.setattr(run_e004_module, "_run", fail_if_called)
    args = Namespace(
        device="cpu",
        task_file=tmp_path / "tasks.jsonl",
        task_bank_calibration_dir=tmp_path / "calibration",
        layers="blocks.2.hook_resid_post",
        feature_selection_modes="top-absolute",
        operations="ablate",
        amplify_factors="2.0",
        control_suite="multi_control",
        ranking_top_k=10,
        eval_top_k=2,
        min_calibrated_per_family=1,
        min_family_consistency=2,
        random_seeds="7,11,13",
        out_root=tmp_path / "matrix",
        force=False,
    )

    payload = run_matrix(args)

    assert payload["completed_cells"] == 1
    assert payload["cells"][0]["resumed_from_existing_artifacts"] is True
    assert calls == []
