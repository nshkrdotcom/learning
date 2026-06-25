from __future__ import annotations

import json
import os
import textwrap

from mechledger.alias import append_alias_record, resolve_run_alias
from mechledger.artifacts import annotate_artifact, attach_artifact
from mechledger.run_capture import capture_command, generate_run_id
from mechledger.sdk import start


def test_run_id_is_portable_and_contains_experiment_and_slug():
    run_id = generate_run_id(experiment_id="E001", purpose="Verify intervention path")

    assert "_e001_verify_intervention_path_" in run_id
    assert len(run_id) <= 120
    assert " " not in run_id
    assert ":" not in run_id


def test_capture_command_writes_run_contract_and_auto_collects_artifacts(tmp_path):
    script = tmp_path / "script.py"
    script.write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path
            artifacts = Path(os.environ["MECHLEDGER_ARTIFACTS_DIR"])
            artifacts.mkdir(parents=True, exist_ok=True)
            (artifacts / "result.json").write_text('{"ok": true}\\n')
            print("hello stdout")
            """
        ),
        encoding="utf-8",
    )

    result = capture_command(
        ["python", str(script)],
        project_root=tmp_path,
        experiment_id="E001",
        run_class="diagnostic",
        purpose="verify intervention path",
    )

    run_dir = tmp_path / ".mechledger" / "runs" / result.run_id
    run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert run_json["status"] == "completed"
    assert run_json["exit_code"] == 0
    assert (run_dir / "heartbeat.json").exists() is False
    assert "hello stdout" in (run_dir / "stdout.txt").read_text(encoding="utf-8")
    assert manifest["artifacts"][0]["review_status"] == "unannotated"
    assert manifest["artifacts"][0]["claim_relevance"] == "none"
    assert (run_dir / "run_ledger_row.csv").exists()


def test_sdk_context_manager_logs_metrics_artifacts_and_exceptions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    artifact = tmp_path / "result.jsonl"
    artifact.write_text('{"metric": 1}\n', encoding="utf-8")

    try:
        with start(experiment="E001", run_class="notebook_exploration", purpose="nb") as run:
            run.log_metric("specificity_gap_mean", 0.123, family="negation")
            run.log_artifact(artifact, claim_relevance="supporting")
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    run_json = json.loads((run.run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_json["status"] == "interrupted"
    assert "specificity_gap_mean" in (run.run_dir / "metrics.jsonl").read_text(encoding="utf-8")
    assert "supporting" in (run.run_dir / "artifacts.jsonl").read_text(encoding="utf-8")


def test_alias_resolution_prefers_latest_and_rejects_ambiguous_prefixes(tmp_path):
    cache = tmp_path / ".mechledger" / "alias_cache.txt"
    append_alias_record(cache, "20260625T120301Z_e001_alpha_abc123", "E001", "alpha")
    append_alias_record(cache, "20260625T120401Z_e001_beta_def456", "E001", "beta")

    assert resolve_run_alias(cache, "latest").run_id == "20260625T120401Z_e001_beta_def456"
    assert resolve_run_alias(cache, "latest:2").run_id == "20260625T120301Z_e001_alpha_abc123"
    assert resolve_run_alias(cache, "beta").run_id == "20260625T120401Z_e001_beta_def456"
    assert resolve_run_alias(cache, "20260625T1203").run_id == "20260625T120301Z_e001_alpha_abc123"

    try:
        resolve_run_alias(cache, "e001")
    except ValueError as error:
        assert "ambiguous" in str(error)
    else:
        raise AssertionError("expected ambiguity")


def test_attach_and_annotate_artifact_records_hash_and_events(tmp_path):
    run_id = "20260625T120301Z_e001_alpha_abc123"
    run_dir = tmp_path / ".mechledger" / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "artifact_manifest.json").write_text('{"artifacts": []}\n', encoding="utf-8")
    path = tmp_path / "results.jsonl"
    path.write_text("{}\n", encoding="utf-8")

    artifact = attach_artifact(run_dir, path, claim_relevance="diagnostic", description="rows")
    updated = annotate_artifact(
        run_dir,
        artifact.artifact_id,
        claim_relevance="supporting",
        description="reviewed rows",
    )

    assert updated.claim_relevance == "supporting"
    assert updated.content_hash_status == "computed"
    assert os.path.isabs(updated.resolved_path)
