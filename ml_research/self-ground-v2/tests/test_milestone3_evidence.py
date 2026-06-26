from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from mechledger.assessments.candidate import assess_run_evidence
from mechledger.assessments.controls import evaluate_matched_controls
from mechledger.assessments.empirical_null import evaluate_empirical_null
from mechledger.assessments.paired_statistic import evaluate_paired_statistic
from mechledger.cli import app

runner = CliRunner()


def init_project(tmp_path: Path, *, decision_status: str = "accepted") -> None:
    result = runner.invoke(app, ["init"], catch_exceptions=False, env={"PWD": str(tmp_path)})
    assert result.exit_code == 0, result.output
    (tmp_path / "research/logs/claim_ledger.md").write_text("# Claim Ledger\n\n", encoding="utf-8")
    write_decision_log(tmp_path, decision_status)


def write_decision_log(tmp_path: Path, d001_status: str = "accepted") -> None:
    (tmp_path / "research/logs/decision_log.md").write_text(
        f"""# Decision Log

## D001 - Evidence waiver review

```yaml
decision_id: D001
status: {d001_status}
affected_experiments: [E999]
affected_claims: []
decision_type: methodology
copilot_session_id: null
```
""",
        encoding="utf-8",
    )


def create_run(
    tmp_path: Path,
    *,
    run_id: str,
    run_class: str,
    metrics: dict[str, Any],
    claim_relevance: str = "required",
) -> Path:
    run_dir = tmp_path / ".mechledger/runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "experiment_id": "E999",
                "run_class": run_class,
                "status": "completed",
                "started_at": "2026-06-25T00:00:00Z",
                "finished_at": "2026-06-25T00:00:01Z",
                "exit_code": 0,
                "command": "python synthetic.py",
                "argv": ["python", "synthetic.py"],
                "purpose": "milestone 3 evidence fixture",
                "hypothesis": "registered metrics are sufficient",
                "model": "pythia-70m",
                "hook_point": "blocks.2.hook_resid_post",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for metric_name, value in metrics.items():
            handle.write(json.dumps({"metric_name": metric_name, "value": value}) + "\n")
    artifact = {
        "artifact_id": "A001",
        "original_path": "artifacts/null_distribution.jsonl",
        "project_relative_path": "artifacts/null_distribution.jsonl",
        "resolved_path": str(tmp_path / "artifacts/null_distribution.jsonl"),
        "claim_relevance": claim_relevance,
        "review_status": "annotated",
        "content_hash": "sha256:test",
        "content_hash_status": "computed",
        "artifact_storage_backend": "git",
        "byte_size": 10,
        "auto_collected": False,
    }
    (tmp_path / "artifacts").mkdir(exist_ok=True)
    (tmp_path / "artifacts/null_distribution.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "artifact_manifest.json").write_text(
        json.dumps({"artifacts": [artifact]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "artifacts.jsonl").write_text(json.dumps(artifact) + "\n", encoding="utf-8")
    for name in ("events.jsonl", "stdout.txt", "stderr.txt"):
        (run_dir / name).write_text("", encoding="utf-8")
    (tmp_path / ".mechledger/alias_cache.txt").write_text(
        f"{run_id}\t2026-06-25T00:00:00Z\tE999\t{run_id}\n", encoding="utf-8"
    )
    return run_dir


def passing_metrics() -> dict[str, Any]:
    return {
        "intended_direction_pass_rate": 1.0,
        "baseline_contrast": 0.5,
        "positive_control_pass_rate": 0.95,
        "random_null_seed_count": 30,
        "null_distribution_path": "artifacts/null_distribution.jsonl",
        "percentile_rank": 0.99,
        "paired_test_name": "wilcoxon",
        "paired_by": "task_id",
        "paired_test_n_pairs": 40,
        "paired_test_p_value": 0.01,
        "effect_direction": "positive",
        "sign_consistency": 0.9,
        "target_delta": 0.4,
        "matched_control_delta": 0.05,
        "specificity_gap": 0.35,
        "top_control_ratio": 0.2,
        "multi_control_min_gap": 0.1,
        "family_min_gap": 0.1,
        "relative_norm_drift": 0.1,
        "nonfinite_rate": 0.0,
        "skip_rate": 0.0,
        "metadata_compatible": True,
    }


def assert_threshold_metadata(condition: Any) -> None:
    assert condition.threshold_source is not None
    assert "threshold" in condition.parameters or "min_" in json.dumps(condition.parameters)
    assert hasattr(condition, "threshold_justification")
    assert hasattr(condition, "threshold_decision_id")


def test_empirical_null_missing_insufficient_and_passing_seed_count() -> None:
    missing = evaluate_empirical_null({})
    assert not missing.conditions["empirical_null_present"].passed
    assert missing.conditions["empirical_null_present"].debt_type == "missing_empirical_null"

    insufficient = evaluate_empirical_null({"random_null_seed_count": 3})
    seed_condition = insufficient.conditions["random_null_seed_count"]
    assert not seed_condition.passed
    assert seed_condition.debt_type == "insufficient_null_seeds"
    assert_threshold_metadata(seed_condition)

    passing = evaluate_empirical_null(
        {
            "random_null_seed_count": 30,
            "null_distribution_path": "artifacts/null_distribution.jsonl",
            "percentile_rank": 0.99,
        },
        percentile_threshold=0.95,
    )
    assert passing.passed
    assert_threshold_metadata(passing.conditions["empirical_null_percentile_rank"])


def test_paired_statistic_missing_insufficient_pairs_and_passing_registration() -> None:
    missing = evaluate_paired_statistic({}, required=True)
    assert not missing.conditions["paired_statistic_present"].passed
    assert missing.conditions["paired_statistic_present"].debt_type == "missing_paired_statistic"

    insufficient = evaluate_paired_statistic(
        {
            "paired_test_name": "sign",
            "paired_by": "task_id",
            "paired_test_n_pairs": 4,
            "paired_test_p_value": 0.01,
            "effect_direction": "positive",
        },
        required=True,
    )
    assert not insufficient.conditions["paired_n_pairs"].passed
    assert_threshold_metadata(insufficient.conditions["paired_n_pairs"])

    passing = evaluate_paired_statistic(
        {
            "paired_test_name": "wilcoxon",
            "paired_by": "task_id",
            "n_pairs": 40,
            "p_value": 0.01,
            "effect_direction": "positive",
            "sign_consistency": 0.85,
        },
        required=True,
    )
    assert passing.passed
    assert_threshold_metadata(passing.conditions["paired_p_value"])


def test_matched_controls_missing_control_dominates_and_positive_specificity_gap() -> None:
    missing = evaluate_matched_controls({}, required=True)
    assert not missing.conditions["matched_controls_present"].passed
    assert missing.conditions["matched_controls_present"].debt_type == "missing_matched_controls"

    dominated = evaluate_matched_controls(
        {"target_delta": 0.1, "matched_control_delta": 0.2, "specificity_gap": -0.1},
        required=True,
    )
    assert not dominated.conditions["specificity_gap_positive"].passed

    passing = evaluate_matched_controls(
        {
            "target_delta": 0.4,
            "matched_control_delta": 0.05,
            "specificity_gap": 0.35,
            "top_control_ratio": 0.2,
            "multi_control_min_gap": 0.1,
            "family_min_gap": 0.1,
        },
        required=True,
    )
    assert passing.passed
    assert_threshold_metadata(passing.conditions["specificity_gap_positive"])


def test_run_class_gating_and_full_passing_serious_run(tmp_path: Path) -> None:
    init_project(tmp_path)
    diagnostic = create_run(
        tmp_path, run_id="RUN_DIAGNOSTIC", run_class="diagnostic", metrics=passing_metrics()
    )
    smoke = create_run(
        tmp_path, run_id="RUN_SMOKE", run_class="smoke_test", metrics=passing_metrics()
    )
    serious = create_run(
        tmp_path,
        run_id="RUN_SERIOUS",
        run_class="serious_evidence_run",
        metrics=passing_metrics(),
    )

    diagnostic_report = assess_run_evidence(diagnostic, project_root=tmp_path)
    assert not diagnostic_report.clean_candidate_support
    assert diagnostic_report.recommended_claim_status != "candidate_claim"
    assert any(debt.debt_type == "diagnostic_run_only" for debt in diagnostic_report.debts)

    smoke_report = assess_run_evidence(smoke, project_root=tmp_path)
    assert not smoke_report.clean_candidate_support
    assert any(debt.debt_type == "smoke_test_only" for debt in smoke_report.debts)

    serious_report = assess_run_evidence(serious, project_root=tmp_path)
    assert serious_report.clean_candidate_support
    assert serious_report.recommended_claim_status == "candidate_claim"


def test_waived_blocking_debt_remains_visible_but_no_longer_blocks(tmp_path: Path) -> None:
    init_project(tmp_path)
    run_dir = create_run(
        tmp_path,
        run_id="RUN_WAIVED_NULL",
        run_class="serious_evidence_run",
        metrics={k: v for k, v in passing_metrics().items() if k != "random_null_seed_count"},
    )
    (run_dir / "scientific_debt_report.json").write_text(
        json.dumps(
            {
                "report_id": "SDR-RUN_WAIVED_NULL",
                "run_id": "RUN_WAIVED_NULL",
                "experiment_id": "E999",
                "generated_at": "2026-06-25T00:00:00Z",
                "evaluated_assessments": ["empirical_null"],
                "threshold_sources": [],
                "clean_candidate_support": False,
                "summary": "previous",
                "debts": [
                    {
                        "debt_id": "DPT006",
                        "debt_type": "missing_empirical_null",
                        "severity": "serious",
                        "claim_id": None,
                        "run_id": "RUN_WAIVED_NULL",
                        "experiment_id": "E999",
                        "evidence_paths": [],
                        "message": "waived for test",
                        "required_resolution": "accepted decision",
                        "status": "waived",
                        "waiver_decision_id": "D001",
                        "created_at": "2026-06-25T00:00:00Z",
                        "resolved_at": "2026-06-25T00:00:00Z",
                        "assessment_id": "empirical_null",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = assess_run_evidence(run_dir, project_root=tmp_path)

    assert report.clean_candidate_support
    waived = [debt for debt in report.debts if debt.debt_id == "DPT006"]
    assert waived and waived[0].status == "waived"
    assert report.waivers_applied == ["DPT006"]


def test_e002_e003_e004_dogfood_metrics_remain_non_candidate(tmp_path: Path) -> None:
    init_project(tmp_path)
    dogfood_cases = {
        "E002": {"target_delta": 0.1, "matched_control_delta": 0.2, "specificity_gap": -0.1},
        "E003": {"target_delta": 0.12, "matched_control_delta": 0.25, "specificity_gap": -0.13},
        "E004": {"specificity_gap": 0.2, "multi_control_min_gap": -0.1, "family_min_gap": -0.2},
    }
    for experiment_id, control_metrics in dogfood_cases.items():
        run_dir = create_run(
            tmp_path,
            run_id=f"RUN_{experiment_id}",
            run_class="serious_evidence_run",
            metrics={**passing_metrics(), **control_metrics},
        )
        report = assess_run_evidence(run_dir, project_root=tmp_path)
        assert not report.clean_candidate_support
        assert report.recommended_claim_status == "failed_or_weakened"


def test_gate_check_writes_evidence_and_debt_reports_for_passing_run(tmp_path: Path) -> None:
    init_project(tmp_path)
    create_run(
        tmp_path,
        run_id="RUN_PASSING",
        run_class="serious_evidence_run",
        metrics=passing_metrics(),
    )

    result = runner.invoke(
        app,
        ["gate", "check", "latest"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    assert "recommended_claim_status: candidate_claim" in result.output
    run_dir = tmp_path / ".mechledger/runs/RUN_PASSING"
    assessment = json.loads((run_dir / "evidence_assessment.json").read_text(encoding="utf-8"))
    assert assessment["schema_version"] == "0.1.0"
    assert assessment["clean_candidate_support"] is True
    assert assessment["recommended_claim_status"] == "candidate_claim"
    assert assessment["conditions"]
    for condition in assessment["conditions"].values():
        assert "threshold_source" in condition
        assert "threshold_justification" in condition
        assert "threshold_decision_id" in condition
    assert "MechLedger assessed registered metadata/artifacts" in (
        run_dir / "evidence_assessment.md"
    ).read_text(encoding="utf-8")
    debt_report = json.loads((run_dir / "scientific_debt_report.json").read_text())
    assert debt_report["clean_candidate_support"] is True


def test_gate_check_blocks_missing_null_then_accepts_waiver(tmp_path: Path) -> None:
    init_project(tmp_path, decision_status="proposed")
    metrics = {k: v for k, v in passing_metrics().items() if k != "random_null_seed_count"}
    create_run(
        tmp_path,
        run_id="RUN_MISSING_NULL",
        run_class="serious_evidence_run",
        metrics=metrics,
    )

    first = runner.invoke(
        app,
        ["gate", "check", "latest"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert first.exit_code == 1, first.output
    assert "missing_empirical_null" in first.output

    proposed = runner.invoke(
        app,
        ["debt", "waive", "DPT006", "--decision", "D001"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert proposed.exit_code == 2

    write_decision_log(tmp_path, "rejected")
    rejected = runner.invoke(
        app,
        ["debt", "waive", "DPT006", "--decision", "D001"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert rejected.exit_code == 2

    write_decision_log(tmp_path, "accepted")
    waived = runner.invoke(
        app,
        ["debt", "waive", "DPT006", "--decision", "D001"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert waived.exit_code == 0, waived.output

    second = runner.invoke(
        app,
        ["gate", "check", "latest"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert second.exit_code == 0, second.output
    debt_payload = json.loads(
        (tmp_path / ".mechledger/runs/RUN_MISSING_NULL/scientific_debt_report.json").read_text()
    )
    debt = next(item for item in debt_payload["debts"] if item["debt_id"] == "DPT006")
    assert debt["status"] == "waived"
    assert debt["waiver_decision_id"] == "D001"
