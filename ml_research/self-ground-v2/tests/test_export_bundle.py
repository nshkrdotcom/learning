from __future__ import annotations

import json
import tarfile
from pathlib import Path

from helpers_project import populate_project, runner

from mechledger.cli import app


def _read_tar(path: Path) -> tuple[dict, set[str]]:
    with tarfile.open(path, "r:gz") as archive:
        names = set(archive.getnames())
        manifest_file = archive.extractfile("manifest.json")
        assert manifest_file is not None
        manifest = json.loads(manifest_file.read().decode("utf-8"))
    return manifest, names


def test_bundle_includes_canonical_files_run_metadata_and_manifest_hashes(tmp_path: Path) -> None:
    populate_project(tmp_path)
    out = tmp_path / "bundles/mechledger_bundle.tar.gz"

    result = runner.invoke(
        app,
        ["export", "bundle", "--out", str(out), "--run", "latest"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    manifest, names = _read_tar(out)
    assert "research/logs/claim_ledger.md" in names
    assert ".mechledger/project.json" in names
    assert ".mechledger/runs/RUN_E001/run.json" in names
    assert ".mechledger/runs/RUN_E001/artifact_manifest.json" in names
    assert "artifacts/result.json" not in names
    assert ".mechledger/index.sqlite" not in names
    assert manifest["included_run_ids"] == ["RUN_E001"]
    assert manifest["redaction_policy"]["redact_env"] is True
    for file_entry in manifest["files"]:
        assert len(file_entry["sha256"]) == 64


def test_bundle_includes_artifact_bytes_only_when_requested_and_refuses_bad_alias(
    tmp_path: Path,
) -> None:
    populate_project(tmp_path)
    out = tmp_path / "bundles/with_artifacts.tar.gz"

    result = runner.invoke(
        app,
        [
            "export",
            "bundle",
            "--out",
            str(out),
            "--run",
            "RUN_E001",
            "--include-artifacts",
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert result.exit_code == 0, result.output
    manifest, names = _read_tar(out)
    assert "artifacts/result.json" in names
    assert manifest["artifact_metadata"][0]["artifact_id"] == "A001"

    bad = runner.invoke(
        app,
        ["export", "bundle", "--out", str(tmp_path / "bad.tar.gz"), "--run", "missing"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert bad.exit_code == 2
    assert "Alias `missing`" in bad.output


def test_bundle_manifest_only_and_zst_guidance(tmp_path: Path) -> None:
    populate_project(tmp_path)
    manifest_out = tmp_path / "bundles/manifest.json"
    manifest_only = runner.invoke(
        app,
        ["export", "bundle", "--out", str(manifest_out), "--manifest-only"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert manifest_only.exit_code == 0, manifest_only.output
    assert json.loads(manifest_out.read_text(encoding="utf-8"))["schema_version"] == "0.1.0"

    zst = runner.invoke(
        app,
        ["export", "bundle", "--out", str(tmp_path / "bundle.tar.zst")],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    if zst.exit_code != 0:
        assert zst.exit_code == 2
        assert "zstd" in zst.output


def test_bundle_manifest_only_accepts_project_relative_output_path(tmp_path: Path) -> None:
    populate_project(tmp_path)

    result = runner.invoke(
        app,
        ["export", "bundle", "--out", "bundles/manifest.json", "--manifest-only"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    assert "bundle: bundles/manifest.json" in result.output
    assert (tmp_path / "bundles/manifest.json").exists()
