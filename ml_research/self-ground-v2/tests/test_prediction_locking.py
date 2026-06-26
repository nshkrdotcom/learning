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


def prediction_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prediction_id": "PRED001",
        "feature_id": "sae_12300",
        "source_examples_path": "research/examples/sae_12300.jsonl",
        "prediction_artifact_path": "research/predictions/sae_12300.json",
        "label_source_model": "gpt-4.1",
        "label_prompt_path": "prompts/explainer_label.md",
        "label_generated_at": "2026-06-25T00:00:00Z",
        "short_label": "negation-sensitive direction feature",
        "predicted_target_direction": "increase",
        "predicted_control_direction": "decrease",
        "predicted_relative_magnitude": "target_gt_control",
        "notes": "human-authored note should be preserved",
    }
    payload.update(overrides)
    return payload


def write_prediction(path: Path, **overrides: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(prediction_payload(**overrides), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def create_run(
    tmp_path: Path,
    *,
    run_id: str = "RUN_SCORE",
    metrics: dict[str, Any] | None = None,
    run_feature_id: str | None = None,
    event_metadata: dict[str, Any] | None = None,
    metric_metadata: dict[str, Any] | None = None,
) -> Path:
    run_dir = tmp_path / ".mechledger/runs" / run_id
    run_dir.mkdir(parents=True)
    run_json: dict[str, Any] = {
        "run_id": run_id,
        "experiment_id": "E002",
        "run_class": "serious_evidence_run",
        "status": "completed",
        "started_at": "2026-06-25T00:00:00Z",
        "finished_at": "2026-06-25T00:00:01Z",
        "exit_code": 0,
        "command": "python intervention.py",
        "argv": ["python", "intervention.py"],
    }
    if run_feature_id is not None:
        run_json["feature_id"] = run_feature_id
    (run_dir / "run.json").write_text(
        json.dumps(run_json, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for metric_name, value in (metrics or {}).items():
            handle.write(
                json.dumps(
                    {
                        "metric_name": metric_name,
                        "value": value,
                        "metadata": metric_metadata or {},
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    event = {
        "timestamp": "2026-06-25T00:00:00Z",
        "event_type": "intervention_logged",
        "message": "intervention metadata logged",
        "metadata": event_metadata or {},
    }
    (run_dir / "events.jsonl").write_text(
        json.dumps(event, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "artifact_manifest.json").write_text(
        json.dumps({"artifacts": []}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "artifacts.jsonl").write_text("", encoding="utf-8")
    (tmp_path / ".mechledger/alias_cache.txt").write_text(
        f"{run_id}\t2026-06-25T00:00:00Z\tE002\t{run_id}\n", encoding="utf-8"
    )
    return run_dir


def test_prediction_lock_writes_hash_status_and_preserves_fields(tmp_path: Path) -> None:
    init_project(tmp_path)
    path = write_prediction(tmp_path / "predictions/sae_12300.json")

    result = runner.invoke(
        app,
        ["prediction", "lock", str(path)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["prediction_id"] == "PRED001"
    assert payload["notes"] == "human-authored note should be preserved"
    assert payload["locked_at"]
    assert len(payload["locked_content_hash"]) == 64
    assert payload["tamper_status"] == "locked_valid"
    assert "newly_locked" in result.output


def test_prediction_lock_hash_is_semantic_and_excludes_mutable_fields(tmp_path: Path) -> None:
    init_project(tmp_path)
    first = write_prediction(tmp_path / "predictions/first.json")
    second = tmp_path / "predictions/second.json"
    second.parent.mkdir(parents=True, exist_ok=True)
    second.write_text(
        json.dumps(
            {
                "tamper_status": "invalidated",
                "sign_match": False,
                "scored_against_run_id": "OLD_RUN",
                "relative_magnitude_match": False,
                "locked_content_hash": "0" * 64,
                "locked_at": "2026-01-01T00:00:00Z",
                **prediction_payload(),
            },
            indent=4,
        )
        + "\n",
        encoding="utf-8",
    )

    first_result = runner.invoke(
        app,
        ["prediction", "lock", str(first)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    second_result = runner.invoke(
        app,
        ["prediction", "lock", str(second), "--force"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert first_result.exit_code == 0, first_result.output
    assert second_result.exit_code == 0, second_result.output
    first_payload = json.loads(first.read_text(encoding="utf-8"))
    second_payload = json.loads(second.read_text(encoding="utf-8"))
    assert first_payload["locked_content_hash"] == second_payload["locked_content_hash"]


def test_prediction_lock_idempotent_and_blocks_modified_after_lock(tmp_path: Path) -> None:
    init_project(tmp_path)
    path = write_prediction(tmp_path / "predictions/sae_12300.json")

    first = runner.invoke(
        app,
        ["prediction", "lock", str(path)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert first.exit_code == 0, first.output
    locked = json.loads(path.read_text(encoding="utf-8"))

    second = runner.invoke(
        app,
        ["prediction", "lock", str(path)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert second.exit_code == 0, second.output
    unchanged = json.loads(path.read_text(encoding="utf-8"))
    assert unchanged["locked_at"] == locked["locked_at"]
    assert unchanged["locked_content_hash"] == locked["locked_content_hash"]
    assert "already_locked" in second.output

    unchanged["short_label"] = "post-hoc edited label"
    path.write_text(json.dumps(unchanged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    modified = runner.invoke(
        app,
        ["prediction", "lock", str(path)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert modified.exit_code == 1, modified.output
    assert "modified_after_lock" in modified.output
    assert json.loads(path.read_text(encoding="utf-8"))["tamper_status"] == "modified_after_lock"

    forced = runner.invoke(
        app,
        ["prediction", "lock", str(path), "--force"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    relocked = json.loads(path.read_text(encoding="utf-8"))
    assert forced.exit_code == 0, forced.output
    assert "force_relocked" in forced.output
    assert relocked["tamper_status"] == "locked_valid"
    assert relocked["locked_at"] != locked["locked_at"]
    assert relocked["locked_content_hash"] != locked["locked_content_hash"]


def test_prediction_lock_invalid_files_fail_with_exit_2(tmp_path: Path) -> None:
    init_project(tmp_path)
    missing = write_prediction(tmp_path / "predictions/missing.json")
    payload = json.loads(missing.read_text(encoding="utf-8"))
    payload.pop("feature_id")
    missing.write_text(json.dumps(payload), encoding="utf-8")

    invalid_enum = write_prediction(
        tmp_path / "predictions/invalid_enum.json",
        predicted_target_direction="larger",
    )
    malformed = tmp_path / "predictions/malformed.json"
    malformed.write_text("{", encoding="utf-8")
    non_mapping = tmp_path / "predictions/list.json"
    non_mapping.write_text("[]\n", encoding="utf-8")

    cases = [
        (missing, "feature_id"),
        (invalid_enum, "predicted_target_direction"),
        (malformed, "malformed JSON"),
        (non_mapping, "JSON object"),
    ]
    for path, message in cases:
        result = runner.invoke(
            app,
            ["prediction", "lock", str(path)],
            catch_exceptions=False,
            env={"PWD": str(tmp_path)},
        )
        assert result.exit_code == 2, result.output
        assert message in result.output


def test_prediction_score_success_uses_alias_and_writes_score_fields(tmp_path: Path) -> None:
    init_project(tmp_path)
    prediction = write_prediction(tmp_path / "research/predictions/sae_12300.json")
    create_run(
        tmp_path,
        metrics={"target_delta": 0.4, "matched_control_delta": -0.1, "specificity_gap": 0.5},
        event_metadata={"features_modified": ["sae_12300"]},
    )
    lock = runner.invoke(
        app,
        ["prediction", "lock", str(prediction)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert lock.exit_code == 0, lock.output

    result = runner.invoke(
        app,
        ["prediction", "score", "PRED001", "--against-run", "latest"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(prediction.read_text(encoding="utf-8"))
    assert payload["scored_against_run_id"] == "RUN_SCORE"
    assert payload["sign_match"] is True
    assert payload["relative_magnitude_match"] is True
    assert payload["tamper_status"] == "locked_valid"
    assert "sign_match: true" in result.output


def test_prediction_score_supports_metric_variants_and_feature_sources(tmp_path: Path) -> None:
    init_project(tmp_path)
    cases = [
        ("event_feature", {"feature_id": "sae_12300"}, {}, None),
        ("metric_feature", {}, {"feature_id": "sae_12300"}, None),
        ("run_feature", {}, {}, "sae_12300"),
    ]
    for run_id, event_metadata, metric_metadata, run_feature_id in cases:
        path = write_prediction(
            tmp_path / f"research/predictions/{run_id}.json",
            prediction_id=f"PRED_{run_id}",
        )
        lock = runner.invoke(
            app,
            ["prediction", "lock", str(path)],
            catch_exceptions=False,
            env={"PWD": str(tmp_path)},
        )
        assert lock.exit_code == 0, lock.output
        create_run(
            tmp_path,
            run_id=run_id,
            metrics={
                "top_target_delta": 0.3,
                "top_control_delta": -0.1,
                "specificity_gap_mean": 0.4,
            },
            event_metadata=event_metadata,
            metric_metadata=metric_metadata,
            run_feature_id=run_feature_id,
        )
        result = runner.invoke(
            app,
            ["prediction", "score", f"PRED_{run_id}", "--against-run", run_id],
            catch_exceptions=False,
            env={"PWD": str(tmp_path)},
        )
        assert result.exit_code == 0, result.output


def test_prediction_score_blocks_state_and_run_evidence_failures(tmp_path: Path) -> None:
    init_project(tmp_path)
    write_prediction(tmp_path / "research/predictions/unlocked.json")
    locked = write_prediction(
        tmp_path / "research/predictions/locked.json",
        prediction_id="PRED_LOCKED",
    )
    lock = runner.invoke(
        app,
        ["prediction", "lock", str(locked)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert lock.exit_code == 0, lock.output
    create_run(
        tmp_path,
        metrics={"target_delta": 0.4, "matched_control_delta": -0.1, "specificity_gap": 0.5},
        event_metadata={"feature_id": "sae_12300"},
    )

    unlocked_result = runner.invoke(
        app,
        ["prediction", "score", "PRED001", "--against-run", "RUN_SCORE"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert unlocked_result.exit_code == 1, unlocked_result.output
    assert "not locked" in unlocked_result.output

    tampered = json.loads(locked.read_text(encoding="utf-8"))
    tampered["predicted_control_direction"] = "increase"
    locked.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tampered_result = runner.invoke(
        app,
        ["prediction", "score", "PRED_LOCKED", "--against-run", "RUN_SCORE"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert tampered_result.exit_code == 1, tampered_result.output
    assert "modified_after_lock" in tampered_result.output

    locked.write_text(
        json.dumps(tampered | {"feature_id": "other_feature"}, indent=2),
        encoding="utf-8",
    )
    relock = runner.invoke(
        app,
        ["prediction", "lock", str(locked), "--force"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert relock.exit_code == 0, relock.output
    mismatch = runner.invoke(
        app,
        ["prediction", "score", "PRED_LOCKED", "--against-run", "RUN_SCORE"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert mismatch.exit_code == 1, mismatch.output
    assert "feature ID mismatch" in mismatch.output

    create_run(
        tmp_path,
        run_id="RUN_NO_METRICS",
        metrics={},
        event_metadata={"feature_id": "other_feature"},
    )
    missing_metric = runner.invoke(
        app,
        ["prediction", "score", "PRED_LOCKED", "--against-run", "RUN_NO_METRICS"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert missing_metric.exit_code == 1, missing_metric.output
    assert "required scoring metric" in missing_metric.output

    create_run(
        tmp_path,
        run_id="RUN_NO_FEATURE",
        metrics={"target_delta": 0.1, "matched_control_delta": 0.0},
        event_metadata={},
    )
    no_feature = runner.invoke(
        app,
        ["prediction", "score", "PRED_LOCKED", "--against-run", "RUN_NO_FEATURE"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert no_feature.exit_code == 1, no_feature.output
    assert "intervention evidence" in no_feature.output


def test_prediction_score_inconsistent_specificity_gap_is_input_error(tmp_path: Path) -> None:
    init_project(tmp_path)
    prediction = write_prediction(tmp_path / "research/predictions/sae_12300.json")
    lock = runner.invoke(
        app,
        ["prediction", "lock", str(prediction)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert lock.exit_code == 0, lock.output
    create_run(
        tmp_path,
        metrics={"target_delta": 0.4, "matched_control_delta": -0.1, "specificity_gap": 0.2},
        event_metadata={"feature_id": "sae_12300"},
    )

    result = runner.invoke(
        app,
        ["prediction", "score", "PRED001", "--against-run", "latest"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 2, result.output
    assert "inconsistent" in result.output


def test_prediction_score_duplicate_and_unknown_ids_are_input_errors(tmp_path: Path) -> None:
    init_project(tmp_path)
    write_prediction(tmp_path / "research/predictions/one.json")
    write_prediction(tmp_path / "predictions/two.json")

    duplicate = runner.invoke(
        app,
        ["prediction", "score", "PRED001", "--against-run", "RUN_SCORE"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert duplicate.exit_code == 2
    assert "Duplicate prediction_id" in duplicate.output

    unknown = runner.invoke(
        app,
        [
            "prediction",
            "score",
            "PRED_MISSING",
            "--against-run",
            "RUN_SCORE",
            "--prediction-dir",
            "research/predictions",
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert unknown.exit_code == 2
    assert "Unknown prediction_id" in unknown.output


def test_prediction_score_unknown_predictions_with_explicit_directory(tmp_path: Path) -> None:
    init_project(tmp_path)
    other_dir = tmp_path / "custom_predictions"
    write_prediction(other_dir / "custom.json", prediction_id="PRED_CUSTOM")
    create_run(
        tmp_path,
        metrics={"target_delta": 0.4, "matched_control_delta": -0.1, "specificity_gap": 0.5},
        event_metadata={"feature_id": "sae_12300"},
    )
    lock = runner.invoke(
        app,
        ["prediction", "lock", str(other_dir / "custom.json")],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert lock.exit_code == 0, lock.output

    result = runner.invoke(
        app,
        [
            "prediction",
            "score",
            "PRED_CUSTOM",
            "--against-run",
            "latest",
            "--prediction-dir",
            str(other_dir),
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
