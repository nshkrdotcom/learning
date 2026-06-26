from __future__ import annotations

import json
from pathlib import Path

from helpers_project import populate_project, runner

from mechledger.cli import app

RECORD_TYPES = [
    "ActivationRecord",
    "CircuitGraphRecord",
    "WeightAnalysisRecord",
    "CrossModelComparisonRecord",
    "FeatureCorrespondenceRecord",
    "TrainingDynamicsRecord",
    "RemoteJobMetadataRecord",
]


def write_record(tmp_path: Path, record_type: str, record_id: str) -> Path:
    path = tmp_path / "research/records" / f"{record_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "record_id": record_id,
                "record_type": record_type,
                "source_paths": ["artifacts/result.json"],
                "linked_runs": ["RUN_E001"],
                "linked_claims": ["C001"],
                "linked_decisions": ["D001"],
                "artifact_paths": ["artifacts/result.json"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_records_validate_each_type_and_list_show_export_metadata(tmp_path: Path) -> None:
    populate_project(tmp_path)
    for index, record_type in enumerate(RECORD_TYPES, start=1):
        path = write_record(tmp_path, record_type, f"REC{index:03d}")
        result = runner.invoke(
            app,
            ["records", "validate", str(path)],
            catch_exceptions=False,
            env={"PWD": str(tmp_path)},
        )
        assert result.exit_code == 0, result.output

    listed = runner.invoke(
        app,
        ["records", "list"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    shown = runner.invoke(
        app,
        ["records", "show", "REC001"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert listed.exit_code == 0 and "REC007" in listed.output
    assert shown.exit_code == 0 and "ActivationRecord" in shown.output

    out = tmp_path / "bundles/ro-crate"
    exported = runner.invoke(
        app,
        ["export", "ro-crate", "--out", str(out)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert exported.exit_code == 0, exported.output
    crate = json.loads((out / "ro-crate-metadata.json").read_text(encoding="utf-8"))
    assert any(
        entity.get("@id") == "research/records/REC001.json#REC001"
        for entity in crate["@graph"]
    )


def test_records_invalid_fixture_fails_precisely(tmp_path: Path) -> None:
    populate_project(tmp_path)
    path = tmp_path / "research/records/bad.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"record_id": "BAD"}), encoding="utf-8")

    result = runner.invoke(
        app,
        ["records", "validate", str(path)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 2
    assert "record_type" in result.output
