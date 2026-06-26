from __future__ import annotations

import json
from pathlib import Path

from helpers_project import populate_project, runner, write_decision_log

from mechledger.cli import app
from mechledger.prediction import lock_prediction
from mechledger.project import find_project, run_ledger_header
from mechledger.workflows import propose_claim


def _records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_integrity_check_detects_prediction_changed_after_lock(tmp_path: Path) -> None:
    populate_project(tmp_path)
    prediction_path = tmp_path / "research/predictions/prediction.json"
    prediction_path.parent.mkdir(parents=True)
    prediction_path.write_text(
        json.dumps(
            {
                "prediction_id": "PRED001",
                "feature_id": "sae_123",
                "source_examples_path": "examples.jsonl",
                "prediction_artifact_path": "research/predictions/prediction.json",
                "label_source_model": "explainer",
                "label_prompt_path": "prompt.md",
                "label_generated_at": "2026-06-25T00:00:00Z",
                "short_label": "original label",
                "predicted_target_direction": "increase",
                "predicted_control_direction": "decrease",
                "predicted_relative_magnitude": "target_gt_control",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    lock_prediction(prediction_path)
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    payload["short_label"] = "edited label"
    prediction_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    result = runner.invoke(app, ["integrity", "check", "--json"], env={"PWD": str(tmp_path)})

    assert result.exit_code == 1, result.output
    rows = json.loads(result.output)
    assert any(row["object_type"] == "prediction" for row in rows)
    assert any(row["consequence"] == "block_scoring" for row in rows)


def test_integrity_check_detects_claim_artifact_decision_draft_and_run_ledger_staleness(
    tmp_path: Path,
) -> None:
    populate_project(tmp_path)
    project = find_project(tmp_path)
    proposal_path = propose_claim(project, "RUN_E001", regenerate=True)
    row = tmp_path / ".mechledger/runs/RUN_E001/run_ledger_row.csv"
    row.write_text(
        run_ledger_header()
        + "\n"
        + "2026-06-25,RUN_E001,abc,phase,purpose,hypothesis,cmd,pythia,hook,,,,,"
        + ",,,baseline,ablate,completed,,specificity_gap=0.3,,artifact_manifest.json,D001\n",
        encoding="utf-8",
    )
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    proposal["supporting_artifact_paths"] = ["artifacts/result.json"]
    proposal_path.write_text(json.dumps(proposal, indent=2, sort_keys=True) + "\n")
    first = runner.invoke(app, ["integrity", "check"], env={"PWD": str(tmp_path)})
    assert first.exit_code == 0, first.output

    artifact = tmp_path / "artifacts/result.json"
    artifact.write_text('{"ok": false}\n', encoding="utf-8")
    claim_ledger = tmp_path / "research/logs/claim_ledger.md"
    claim_ledger.write_text(
        claim_ledger.read_text(encoding="utf-8").replace(
            "preliminary evidence", "edited preliminary evidence"
        ),
        encoding="utf-8",
    )
    write_decision_log(tmp_path, status="proposed")
    draft = tmp_path / "research/paper/draft.md"
    draft.write_text("Unknown claim. [CLAIM:C999]\n", encoding="utf-8")
    row.write_text(row.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")

    result = runner.invoke(app, ["integrity", "check"], env={"PWD": str(tmp_path)})

    assert result.exit_code == 1, result.output
    record_path = tmp_path / ".mechledger/integrity/tamper_records.jsonl"
    rows = _records(record_path)
    object_types = {row["object_type"] for row in rows}
    assert {
        "artifact",
        "claim_yaml_block",
        "decision_status",
        "draft_claim_tag",
        "run_ledger_proposal",
    } <= object_types
    second = runner.invoke(app, ["integrity", "check"], env={"PWD": str(tmp_path)})
    assert second.exit_code == 1
    assert len(_records(record_path)) == len(rows)


def test_integrity_status_surfaces_unresolved_tamper_records(tmp_path: Path) -> None:
    populate_project(tmp_path)
    draft = tmp_path / "research/paper/draft.md"
    draft.write_text("Unknown claim. [CLAIM:C999]\n", encoding="utf-8")
    runner.invoke(app, ["integrity", "check"], env={"PWD": str(tmp_path)})

    result = runner.invoke(app, ["status"], env={"PWD": str(tmp_path)})

    assert result.exit_code == 0, result.output
    assert "Integrity:" in result.output
    assert "unresolved tamper records: 1" in result.output


def test_integrity_resolve_requires_accepted_decision(tmp_path: Path) -> None:
    populate_project(tmp_path)
    draft = tmp_path / "research/paper/draft.md"
    draft.write_text("Unknown claim. [CLAIM:C999]\n", encoding="utf-8")
    runner.invoke(app, ["integrity", "check"], env={"PWD": str(tmp_path)})
    tamper_id = _records(tmp_path / ".mechledger/integrity/tamper_records.jsonl")[0][
        "tamper_id"
    ]
    write_decision_log(tmp_path, status="proposed")

    bad = runner.invoke(
        app,
        [
            "integrity",
            "resolve",
            str(tamper_id),
            "--decision",
            "D001",
            "--status",
            "waived",
            "--note",
            "reviewed",
        ],
        env={"PWD": str(tmp_path)},
    )
    assert bad.exit_code == 2
    assert "accepted decision" in bad.output

    write_decision_log(tmp_path, status="accepted")
    good = runner.invoke(
        app,
        [
            "integrity",
            "resolve",
            str(tamper_id),
            "--decision",
            "D001",
            "--status",
            "waived",
            "--note",
            "reviewed",
        ],
        env={"PWD": str(tmp_path)},
    )
    assert good.exit_code == 0, good.output
    record = _records(tmp_path / ".mechledger/integrity/tamper_records.jsonl")[0]
    assert record["resolution_status"] == "waived"
    assert record["resolution_decision_id"] == "D001"


def test_stale_claim_proposal_still_refuses_unsafe_apply(tmp_path: Path) -> None:
    populate_project(tmp_path)
    project = find_project(tmp_path)
    propose_claim(project, "RUN_E001", regenerate=True)
    claim_ledger = tmp_path / "research/logs/claim_ledger.md"
    claim_ledger.write_text(
        claim_ledger.read_text(encoding="utf-8").replace(
            "preliminary evidence", "changed evidence"
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["claim", "review", "RUN_E001", "--apply", "--yes"],
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 2
    assert "claim proposal is stale" in result.output
