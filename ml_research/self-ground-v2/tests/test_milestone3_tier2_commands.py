from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from mechledger.cli import app

runner = CliRunner()


def init_project(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init"], catch_exceptions=False, env={"PWD": str(tmp_path)})
    assert result.exit_code == 0, result.output
    (tmp_path / "research/logs/decision_log.md").write_text("# Decision Log\n\n", encoding="utf-8")
    (tmp_path / "research/logs/claim_ledger.md").write_text("# Claim Ledger\n\n", encoding="utf-8")


def create_run(
    tmp_path: Path,
    *,
    run_id: str = "RUN_TIER2",
    metrics: dict[str, Any] | None = None,
) -> Path:
    run_dir = tmp_path / ".mechledger/runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "experiment_id": "E001",
                "run_class": "serious_evidence_run",
                "status": "completed",
                "started_at": "2026-06-25T00:00:00Z",
                "finished_at": "2026-06-25T00:00:01Z",
                "exit_code": 0,
                "command": "python researcher_owned.py",
                "argv": ["python", "researcher_owned.py"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_metrics(run_dir, metrics or {})
    (run_dir / "artifact_manifest.json").write_text(
        json.dumps({"artifacts": []}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name in ("artifacts.jsonl", "events.jsonl", "stdout.txt", "stderr.txt"):
        (run_dir / name).write_text("", encoding="utf-8")
    (tmp_path / ".mechledger/alias_cache.txt").write_text(
        f"{run_id}\t2026-06-25T00:00:00Z\tE001\t{run_id}\n", encoding="utf-8"
    )
    return run_dir


def write_metrics(run_dir: Path, metrics: dict[str, Any]) -> None:
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for metric_name, value in metrics.items():
            handle.write(json.dumps({"metric_name": metric_name, "value": value}) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def passing_gate_metrics() -> dict[str, Any]:
    return {
        "intended_direction_pass_rate": 1.0,
        "baseline_contrast": 0.5,
        "positive_control_pass_rate": 0.95,
        "random_null_seed_count": 30,
        "null_distribution_path": "outputs/null_distribution.jsonl",
        "percentile_rank": 0.99,
        "paired_test_name": "sign",
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


def test_calibration_check_writes_filtered_report_and_resolves_alias(tmp_path: Path) -> None:
    init_project(tmp_path)
    run_dir = create_run(
        tmp_path,
        metrics={
            "intended_direction_pass_rate": 1.0,
            "baseline_contrast": 0.4,
            "positive_control_pass_rate": 0.95,
        },
    )

    result = runner.invoke(
        app,
        ["calibration", "check", "latest"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads((run_dir / "calibration_check.json").read_text(encoding="utf-8"))
    assert payload["run_id"] == "RUN_TIER2"
    assert payload["assessment_ids"] == ["baseline_calibration", "positive_control"]
    assert payload["clean"] is True
    assert set(payload["conditions"]) == {
        "baseline_calibration_recorded",
        "positive_control_pass_rate",
    }
    assert all(
        condition["threshold_source"] for condition in payload["conditions"].values()
    )
    markdown = (run_dir / "calibration_check.md").read_text(encoding="utf-8")
    assert "baseline_calibration_recorded" in markdown
    assert "positive_control_pass_rate" in markdown
    assert "threshold_source" in markdown
    assert "unjustified_threshold_default" in markdown


def test_calibration_check_blocks_failed_positive_control(tmp_path: Path) -> None:
    init_project(tmp_path)
    create_run(
        tmp_path,
        metrics={
            "intended_direction_pass_rate": 1.0,
            "baseline_contrast": 0.4,
            "positive_control_pass_rate": 0.2,
        },
    )

    result = runner.invoke(
        app,
        ["calibration", "check", "RUN_TIER2"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 1, result.output
    assert "failed_positive_control" in result.output


def test_telemetry_check_writes_filtered_report_and_blocks_nonfinite(tmp_path: Path) -> None:
    init_project(tmp_path)
    run_dir = create_run(
        tmp_path,
        metrics={
            "relative_norm_drift": 0.1,
            "decoded_delta_norm_ratio": 0.2,
            "nonfinite_rate": 0.1,
            "skip_rate": 0.0,
            "reconstruction_mse": 0.01,
            "reconstruction_l2_relative": 0.03,
            "selected_feature_nonzero_fraction": 0.7,
        },
    )

    result = runner.invoke(
        app,
        ["telemetry", "check", "latest"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 1, result.output
    assert "nonfinite_rows" in result.output
    payload = json.loads((run_dir / "telemetry_check.json").read_text(encoding="utf-8"))
    assert payload["assessment_ids"] == ["telemetry"]
    assert set(payload["conditions"]) == {"relative_norm_drift", "nonfinite_rate", "skip_rate"}
    assert all(
        condition["threshold_source"] for condition in payload["conditions"].values()
    )
    markdown = (run_dir / "telemetry_check.md").read_text(encoding="utf-8")
    assert "threshold_source" in markdown
    assert "nonfinite_rate" in markdown


def test_telemetry_check_exits_zero_for_passing_and_nonblocking_debt(tmp_path: Path) -> None:
    init_project(tmp_path)
    run_dir = create_run(
        tmp_path,
        metrics={
            "relative_norm_drift": 0.1,
            "nonfinite_rate": 0.0,
            "skip_rate": 0.0,
        },
    )

    passing = runner.invoke(
        app,
        ["telemetry", "check", "latest"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert passing.exit_code == 0, passing.output
    payload = json.loads((run_dir / "telemetry_check.json").read_text(encoding="utf-8"))
    assert payload["clean"] is True

    write_metrics(
        run_dir,
        {
            "relative_norm_drift": 0.6,
            "nonfinite_rate": 0.0,
            "skip_rate": 0.0,
        },
    )
    nonblocking = runner.invoke(
        app,
        ["telemetry", "check", "RUN_TIER2"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert nonblocking.exit_code == 0, nonblocking.output
    assert "high_norm_drift" in nonblocking.output


def test_null_plan_creates_yaml_refuses_overwrite_and_force_replaces(tmp_path: Path) -> None:
    init_project(tmp_path)

    first = runner.invoke(
        app,
        [
            "null",
            "run",
            "--plan",
            "--experiment",
            "E001",
            "--feature-set-size",
            "20",
            "--seeds",
            "30",
            "--sampling",
            "density_matched",
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert first.exit_code == 0, first.output
    plan_path = tmp_path / "research/experiments/E001_null_plan.yaml"
    plan_text = plan_path.read_text(encoding="utf-8")
    assert "experiment_id: E001" in plan_text
    assert "feature_set_size: 20" in plan_text
    assert "seed_count: 30" in plan_text
    assert "sampling_method: density_matched" in plan_text
    assert "exclude_feature_ids: []" in plan_text
    assert "output_metric: specificity_gap" in plan_text
    assert "planned_output_artifact: runs/null/E001_null_distribution.jsonl" in plan_text

    refused = runner.invoke(
        app,
        [
            "null",
            "run",
            "--plan",
            "--experiment",
            "E001",
            "--feature-set-size",
            "21",
            "--seeds",
            "30",
            "--sampling",
            "density_matched",
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert refused.exit_code == 2
    assert "already exists" in refused.output

    forced = runner.invoke(
        app,
        [
            "null",
            "run",
            "--plan",
            "--experiment",
            "E001",
            "--feature-set-size",
            "21",
            "--seeds",
            "31",
            "--sampling",
            "score_matched",
            "--force",
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert forced.exit_code == 0, forced.output
    forced_text = plan_path.read_text(encoding="utf-8")
    assert "feature_set_size: 21" in forced_text
    assert "seed_count: 31" in forced_text
    assert "sampling_method: score_matched" in forced_text


def test_null_plan_validates_positive_counts_and_sampling(tmp_path: Path) -> None:
    init_project(tmp_path)

    result = runner.invoke(
        app,
        [
            "null",
            "run",
            "--plan",
            "--experiment",
            "E001",
            "--feature-set-size",
            "0",
            "--seeds",
            "30",
            "--sampling",
            "density_matched",
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 2
    assert "feature-set-size" in result.output

    invalid_sampling = runner.invoke(
        app,
        [
            "null",
            "run",
            "--plan",
            "--experiment",
            "E001",
            "--feature-set-size",
            "20",
            "--seeds",
            "30",
            "--sampling",
            "nearest_neighbor",
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert invalid_sampling.exit_code == 2
    assert "sampling" in invalid_sampling.output


def test_null_register_attaches_artifact_logs_metrics_and_reports_insufficient(
    tmp_path: Path,
) -> None:
    init_project(tmp_path)
    run_dir = create_run(tmp_path)
    null_path = tmp_path / "outputs/null_distribution.jsonl"
    null_path.parent.mkdir()
    null_path.write_text('{"seed": 1, "specificity_gap_mean": 0.1}\n', encoding="utf-8")
    null_arg = Path("outputs/null_distribution.jsonl")

    result = runner.invoke(
        app,
        [
            "null",
            "run",
            "--register",
            "latest",
            "--null-distribution",
            str(null_arg),
            "--metric",
            "specificity_gap_mean",
            "--seed-count",
            "3",
            "--percentile-rank",
            "0.9",
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 1, result.output
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    metric_values = {row["metric_name"]: row["value"] for row in metrics}
    assert metric_values["random_null_seed_count"] == 3
    assert metric_values["null_distribution_path"] == str(null_arg)
    assert metric_values["null_metric"] == "specificity_gap_mean"
    assert metric_values["percentile_rank"] == 0.9
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["artifacts"]) == 1
    assert manifest["artifacts"][0]["claim_relevance"] == "required"
    payload = json.loads((run_dir / "null_check.json").read_text(encoding="utf-8"))
    assert payload["assessment_ids"] == ["empirical_null"]
    assert payload["clean"] is False
    assert (run_dir / "scientific_debt_report.json").exists()

    duplicate = runner.invoke(
        app,
        [
            "null",
            "run",
            "--register",
            "RUN_TIER2",
            "--null-distribution",
            str(null_arg),
            "--metric",
            "specificity_gap_mean",
            "--seed-count",
            "3",
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert duplicate.exit_code == 2
    assert "already registered" in duplicate.output


def test_null_register_sufficient_exits_zero(tmp_path: Path) -> None:
    init_project(tmp_path)
    create_run(tmp_path)
    null_path = tmp_path / "null_distribution.jsonl"
    null_path.write_text('{"seed": 1}\n', encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "null",
            "run",
            "--register",
            "RUN_TIER2",
            "--null-distribution",
            str(null_path),
            "--metric",
            "specificity_gap_mean",
            "--seed-count",
            "30",
            "--percentile-rank",
            "0.99",
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(
        (tmp_path / ".mechledger/runs/RUN_TIER2/null_check.json").read_text(encoding="utf-8")
    )
    assert payload["clean"] is True


def test_stats_paired_test_registers_result_metrics_artifact_and_report(tmp_path: Path) -> None:
    init_project(tmp_path)
    run_dir = create_run(tmp_path)
    result_path = tmp_path / "paired_test.json"
    result_arg = Path("paired_test.json")
    result_path.write_text(
        json.dumps(
            {
                "run_id": "RUN_TIER2",
                "paired_by": "task_id",
                "metric": "specificity_gap",
                "test": "sign",
                "n_pairs": 40,
                "p_value": 0.01,
                "effect_direction": "positive",
                "sign_consistency": 0.9,
                "threshold_source": "tool_default",
                "threshold_justification": None,
                "threshold_decision_id": None,
                "input_artifact_path": "outputs/per_task_results.jsonl",
                "output_artifact_path": "paired_test.json",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["stats", "paired-test", "latest", "--register", str(result_arg)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    copied = json.loads((run_dir / "paired_test.json").read_text(encoding="utf-8"))
    assert copied["run_id"] == "RUN_TIER2"
    assert (run_dir / "paired_test.md").exists()
    metric_values = {
        row["metric_name"]: row["value"] for row in read_jsonl(run_dir / "metrics.jsonl")
    }
    assert metric_values["paired_test_name"] == "sign"
    assert metric_values["paired_by"] == "task_id"
    assert metric_values["paired_test_n_pairs"] == 40
    assert metric_values["paired_test_p_value"] == 0.01
    assert metric_values["effect_direction"] == "positive"
    assert metric_values["sign_consistency"] == 0.9
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["artifacts"]) == 1
    assert manifest["artifacts"][0]["claim_relevance"] == "required"
    assert (run_dir / "scientific_debt_report.json").exists()


def test_stats_paired_test_rejects_mismatched_run_id(tmp_path: Path) -> None:
    init_project(tmp_path)
    create_run(tmp_path)
    result_path = tmp_path / "paired_test.json"
    result_path.write_text(
        json.dumps(
            {
                "run_id": "OTHER_RUN",
                "paired_by": "task_id",
                "metric": "specificity_gap",
                "test": "sign",
                "n_pairs": 40,
                "p_value": 0.01,
                "effect_direction": "positive",
                "sign_consistency": 0.9,
                "threshold_source": "tool_default",
                "threshold_justification": None,
                "threshold_decision_id": None,
                "input_artifact_path": "outputs/per_task_results.jsonl",
                "output_artifact_path": "paired_test.json",
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["stats", "paired-test", "RUN_TIER2", "--register", str(result_path)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 2
    assert "does not match" in result.output


def test_sdk_stats_compute_sign_test_and_write_result(tmp_path: Path) -> None:
    import mechledger as ml

    rows = tmp_path / "per_task_results.jsonl"
    rows.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "t1", "specificity_gap": 0.4}),
                json.dumps({"task_id": "t2", "specificity_gap": 0.2}),
                json.dumps({"task_id": "t3", "specificity_gap": -0.1}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = ml.stats.compute_paired_test(
        rows,
        paired_by="task_id",
        metric="specificity_gap",
        test="sign",
    )
    assert result["paired_by"] == "task_id"
    assert result["metric"] == "specificity_gap"
    assert result["test"] == "sign"
    assert result["n_pairs"] == 3
    assert result["effect_direction"] == "positive"
    assert result["sign_consistency"] == 2 / 3
    assert result["threshold_source"] == "tool_default"

    output = tmp_path / "paired_test.json"
    ml.stats.write_paired_test_result(result, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == result


def test_sdk_stats_non_sign_tests_are_register_only(tmp_path: Path) -> None:
    import mechledger as ml

    rows = tmp_path / "per_task_results.jsonl"
    rows.write_text(
        json.dumps({"task_id": "t1", "specificity_gap": 0.4}) + "\n",
        encoding="utf-8",
    )

    for test_name in ("wilcoxon", "permutation"):
        try:
            ml.stats.compute_paired_test(
                rows,
                paired_by="task_id",
                metric="specificity_gap",
                test=test_name,
            )
        except NotImplementedError as exc:
            assert "does not vendor numerical stacks" in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"{test_name} should be register-only")


def test_gate_check_behavior_remains_unchanged_after_tier2_commands(tmp_path: Path) -> None:
    init_project(tmp_path)
    run_dir = create_run(tmp_path, metrics=passing_gate_metrics())

    result = runner.invoke(
        app,
        ["gate", "check", "latest"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    assert "recommended_claim_status: candidate_claim" in result.output
    evidence = json.loads((run_dir / "evidence_assessment.json").read_text(encoding="utf-8"))
    assert evidence["clean_candidate_support"] is True
    assert evidence["recommended_claim_status"] == "candidate_claim"
