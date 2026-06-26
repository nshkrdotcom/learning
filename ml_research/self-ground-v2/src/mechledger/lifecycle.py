from __future__ import annotations

import gzip
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from mechledger.alias import resolve_run_id
from mechledger.inspection import sha256_file, write_json
from mechledger.project import Project, now_utc
from mechledger.run_auditor import append_event, write_run_json

RUN_BUNDLE_FILES = [
    "run.json",
    "events.jsonl",
    "metrics.jsonl",
    "artifacts.jsonl",
    "artifact_manifest.json",
    "resource_usage.json",
    "command.txt",
    "environment.json",
    "git.json",
    "summary.json",
    "scientific_debt_report.json",
]


def pin_run(project: Project, run_alias: str) -> tuple[str, bool]:
    run_id = resolve_run_id(project, run_alias)
    run_dir = project.runs_dir / run_id
    run_json = run_dir / "run.json"
    if not run_json.exists():
        raise FileNotFoundError(f"Run is missing run.json: {run_json}")
    payload = _read_json(run_json)
    if payload.get("pinned") is True:
        return run_id, False
    payload["pinned"] = True
    write_run_json(run_dir, payload)
    append_event(run_dir, "run_pinned", "run pinned for retention")
    return run_id, True


def write_run_bundle(
    project: Project,
    run_alias: str,
    out: Path,
    *,
    include_artifacts: bool = True,
) -> tuple[Path, dict[str, Any]]:
    run_id = resolve_run_id(project, run_alias)
    run_dir = project.runs_dir / run_id
    if not (run_dir / "run.json").exists():
        raise FileNotFoundError(f"Run is missing run.json: {run_dir}")
    files = _run_bundle_files(project, run_id, include_artifacts=include_artifacts)
    manifest = {
        "schema_version": "0.1.0",
        "run_id": run_id,
        "created_by": "mechledger",
        "files": [
            {"path": dest, "sha256": sha256_file(source)} for dest, source in files
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.name.endswith(".tar.gz"):
        _write_tar_gz(out, files, manifest)
    elif out.name.endswith(".tar.zst"):
        _write_tar_zst(out, files, manifest)
    else:
        raise ValueError("Run bundle --out must end in .tar.gz or .tar.zst.")
    return out, manifest


def garbage_collect(
    project: Project,
    *,
    keep_last: int,
    keep_pinned: bool = True,
    archive_dir: Path | None = None,
    yes: bool = False,
    allow_remove_all_unpinned: bool = False,
) -> dict[str, Any]:
    if keep_last < 0:
        raise ValueError("--keep-last must be >= 0.")
    if keep_last == 0 and not allow_remove_all_unpinned:
        raise ValueError(
            "--keep-last 0 requires --allow-remove-all-unpinned to avoid accidental removal."
        )
    if archive_dir is not None and archive_dir.exists() and not archive_dir.is_dir():
        raise ValueError(f"Archive path is not a directory: {archive_dir}")
    runs = _run_retention_rows(project)
    newest_kept = {
        row["run_id"] for row in sorted(runs, key=_run_sort_key, reverse=True)[:keep_last]
    }
    planned = []
    for row in runs:
        if row["run_id"] in newest_kept:
            continue
        if row["pinned"]:
            continue
        planned.append(row)
    manifest = {
        "generated_at": now_utc(),
        "dry_run": not yes,
        "keep_last": keep_last,
        "keep_pinned": keep_pinned,
        "planned_remove_run_ids": [row["run_id"] for row in planned],
        "removed_run_ids": [],
        "archived_run_ids": [],
        "archives": [],
        "run_hashes": {
            row["run_id"]: _directory_hash(row["run_dir"]) for row in planned
        },
    }
    if yes:
        if archive_dir is not None:
            archive_dir.mkdir(parents=True, exist_ok=True)
            for row in planned:
                archive_path = archive_dir / f"{row['run_id']}.tar.gz"
                write_run_bundle(project, row["run_id"], archive_path)
                manifest["archived_run_ids"].append(row["run_id"])
                manifest["archives"].append(str(archive_path))
        for row in planned:
            shutil.rmtree(row["run_dir"])
            manifest["removed_run_ids"].append(row["run_id"])
    write_json(project.mechledger_dir / "gc_manifest.json", manifest)
    return manifest


def _run_bundle_files(
    project: Project, run_id: str, *, include_artifacts: bool
) -> list[tuple[str, Path]]:
    run_dir = project.runs_dir / run_id
    files: list[tuple[str, Path]] = []
    for name in RUN_BUNDLE_FILES:
        path = run_dir / name
        if path.exists() and path.is_file():
            files.append((_rel(project, path), path))
    manifest_path = run_dir / "artifact_manifest.json"
    if include_artifacts and manifest_path.exists():
        manifest = _read_json(manifest_path)
        for artifact in manifest.get("artifacts") or []:
            for key in ("resolved_path", "project_relative_path", "placeholder_path"):
                raw = artifact.get(key)
                if not raw:
                    continue
                path = Path(str(raw))
                if not path.is_absolute():
                    path = project.root / path
                if path.exists() and path.is_file() and _inside_project(project, path):
                    files.append((_rel(project, path), path))
                    break
    return sorted({dest: source for dest, source in files}.items())


def _run_retention_rows(project: Project) -> list[dict[str, Any]]:
    rows = []
    for run_json in sorted(project.runs_dir.glob("*/run.json")):
        payload = _read_json(run_json)
        rows.append(
            {
                "run_id": str(payload.get("run_id") or run_json.parent.name),
                "started_at": str(payload.get("started_at") or ""),
                "pinned": bool(payload.get("pinned")),
                "run_dir": run_json.parent,
            }
        )
    return rows


def _run_sort_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("started_at") or ""), str(row.get("run_id") or ""))


def _write_tar_gz(out: Path, files: list[tuple[str, Path]], manifest: dict[str, Any]) -> None:
    with out.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as archive:
                _add_bytes(archive, "manifest.json", _json_bytes(manifest))
                for dest, source in files:
                    _add_bytes(archive, dest, source.read_bytes())


def _write_tar_zst(out: Path, files: list[tuple[str, Path]], manifest: dict[str, Any]) -> None:
    zstd = shutil.which("zstd")
    if not zstd:
        raise ValueError(
            "Writing .tar.zst requires the `zstd` command-line tool; use .tar.gz "
            "or install zstd."
        )
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as handle:
        temp_tar = Path(handle.name)
    try:
        with tarfile.open(temp_tar, mode="w") as archive:
            _add_bytes(archive, "manifest.json", _json_bytes(manifest))
            for dest, source in files:
                _add_bytes(archive, dest, source.read_bytes())
        subprocess.run([zstd, "-q", "-f", str(temp_tar), "-o", str(out)], check=True)
    finally:
        temp_tar.unlink(missing_ok=True)


def _add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, io.BytesIO(data))


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _directory_hash(path: Path) -> str:
    digest = json.dumps(
        [
            {"path": item.relative_to(path).as_posix(), "sha256": sha256_file(item)}
            for item in sorted(path.rglob("*"))
            if item.is_file()
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    import hashlib

    return hashlib.sha256(digest.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: JSON root must be an object.")
    return payload


def _inside_project(project: Project, path: Path) -> bool:
    try:
        path.resolve().relative_to(project.root.resolve())
    except ValueError:
        return False
    return True


def _rel(project: Project, path: Path) -> str:
    return path.resolve().relative_to(project.root.resolve()).as_posix()
