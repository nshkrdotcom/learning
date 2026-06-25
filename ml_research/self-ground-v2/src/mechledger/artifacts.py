from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from mechledger.io import append_jsonl, read_json, sha256_file, utc_now, write_json
from mechledger.models import ArtifactRecord


def attach_artifact(
    run_dir: str | Path,
    path: str | Path,
    *,
    artifact_type: str | None = None,
    claim_relevance: str = "none",
    description: str | None = None,
    allow_missing: bool = False,
    storage_backend: str = "git",
) -> ArtifactRecord:
    run_dir = Path(run_dir)
    original = str(path)
    resolved = _resolve_path(path, run_dir)
    exists = resolved.exists()
    if not exists and not allow_missing:
        raise FileNotFoundError(str(resolved))
    manifest = _read_manifest(run_dir)
    artifact_id = _next_artifact_id(manifest)
    record = ArtifactRecord(
        artifact_id=artifact_id,
        path=_projectish_path(resolved, run_dir),
        original_path=original,
        resolved_path=str(resolved),
        artifact_type=artifact_type or _infer_type(resolved),
        content_hash=sha256_file(resolved) if exists else None,
        content_hash_status="computed" if exists else "external_unverified",
        artifact_storage_backend=storage_backend,  # type: ignore[arg-type]
        byte_size=resolved.stat().st_size if exists else None,
        claim_relevance=claim_relevance,  # type: ignore[arg-type]
        review_status="annotated"
        if claim_relevance != "none" and exists
        else ("missing" if not exists else "unannotated"),
        description=description,
    )
    manifest["artifacts"].append(record.to_dict())
    write_json(run_dir / "artifact_manifest.json", manifest)
    append_jsonl(
        run_dir / "artifacts.jsonl",
        {
            **record.to_dict(),
            "timestamp": utc_now(),
            "event_type": "artifact_registered",
        },
    )
    append_jsonl(
        run_dir / "events.jsonl",
        {
            "timestamp": utc_now(),
            "event_type": "artifact_registered",
            "message": f"Registered artifact {artifact_id}",
            "metadata": {"artifact_id": artifact_id},
        },
    )
    return record


def annotate_artifact(
    run_dir: str | Path,
    artifact_id: str,
    *,
    claim_relevance: str,
    description: str | None = None,
) -> ArtifactRecord:
    run_dir = Path(run_dir)
    manifest = _read_manifest(run_dir)
    for index, row in enumerate(manifest["artifacts"]):
        if row.get("artifact_id") == artifact_id:
            row = dict(row)
            row["claim_relevance"] = claim_relevance
            row["review_status"] = "annotated"
            if description is not None:
                row["description"] = description
            manifest["artifacts"][index] = row
            write_json(run_dir / "artifact_manifest.json", manifest)
            append_jsonl(
                run_dir / "artifacts.jsonl",
                {
                    **row,
                    "timestamp": utc_now(),
                    "event_type": "artifact_annotated",
                },
            )
            return ArtifactRecord(**_artifact_kwargs(row))
    raise KeyError(f"artifact {artifact_id} not found")


def auto_collect_artifacts(run_dir: str | Path) -> list[ArtifactRecord]:
    run_dir = Path(run_dir)
    artifacts_dir = run_dir / "artifacts"
    if not artifacts_dir.exists():
        return []
    manifest = _read_manifest(run_dir)
    known = {row["resolved_path"] for row in manifest["artifacts"]}
    collected: list[ArtifactRecord] = []
    for path in sorted(item for item in artifacts_dir.rglob("*") if item.is_file()):
        if str(path.resolve()) in known:
            continue
        record = attach_artifact(
            run_dir,
            path,
            claim_relevance="none",
            description=None,
            storage_backend="git",
        )
        append_jsonl(
            run_dir / "events.jsonl",
            {
                "timestamp": utc_now(),
                "event_type": "artifact_auto_collected",
                "message": f"Auto-collected artifact {record.artifact_id}",
                "metadata": {"artifact_id": record.artifact_id},
            },
        )
        collected.append(record)
    return collected


def _read_manifest(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    path = run_dir / "artifact_manifest.json"
    if not path.exists():
        return {"artifacts": []}
    data = read_json(path)
    if "artifacts" not in data or not isinstance(data["artifacts"], list):
        return {"artifacts": []}
    return data  # type: ignore[return-value]


def _next_artifact_id(manifest: dict[str, list[dict[str, Any]]]) -> str:
    return f"A{len(manifest['artifacts']) + 1:03d}"


def _resolve_path(path: str | Path, run_dir: Path) -> Path:
    raw = str(path)
    if raw.startswith("//"):
        project_root = run_dir.parents[2]
        return (project_root / raw[2:]).resolve()
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def _projectish_path(path: Path, run_dir: Path) -> str:
    project_root = run_dir.parents[2]
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _infer_type(path: Path) -> str:
    suffix = path.suffix.lstrip(".")
    if suffix:
        return suffix
    guessed = mimetypes.guess_type(path.name)[0]
    return guessed or "file"


def _artifact_kwargs(row: dict[str, Any]) -> dict[str, Any]:
    fields = ArtifactRecord.__dataclass_fields__.keys()
    return {key: row.get(key) for key in fields}
