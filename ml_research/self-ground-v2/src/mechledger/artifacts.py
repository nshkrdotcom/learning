from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any

from mechledger.project import Project, now_utc


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "artifact_manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"artifacts": []}


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    manifest["artifacts"] = sorted(
        manifest.get("artifacts", []), key=lambda item: item["artifact_id"]
    )
    (run_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def register_artifact(
    project: Project,
    run_id: str,
    path: Path,
    *,
    artifact_type: str | None = None,
    claim_relevance: str = "none",
    description: str | None = None,
    allow_missing: bool = False,
    auto_collected: bool = False,
    storage_backend: str | None = None,
) -> dict[str, Any]:
    run_dir = project.runs_dir / run_id
    manifest = load_manifest(run_dir)
    artifact_id = next_artifact_id(manifest)
    resolved = resolve_artifact_path(project, path)
    exists = resolved.exists()
    if not exists and not allow_missing:
        raise FileNotFoundError(f"Artifact path does not exist: {path}")
    artifact = {
        "artifact_id": artifact_id,
        "original_path": str(path),
        "resolved_path": str(resolved),
        "project_relative_path": _relative_to_project(project, resolved),
        "artifact_type": artifact_type or _artifact_type(resolved),
        "content_hash": sha256_file(resolved) if exists and resolved.is_file() else None,
        "content_hash_status": "computed"
        if exists and resolved.is_file()
        else "external_unverified",
        "artifact_storage_backend": storage_backend or ("git" if exists else "external"),
        "claim_relevance": claim_relevance,
        "review_status": "unannotated" if auto_collected else "annotated" if exists else "missing",
        "description": description,
        "byte_size": resolved.stat().st_size if exists and resolved.is_file() else None,
        "registered_at": now_utc(),
        "auto_collected": auto_collected,
    }
    manifest.setdefault("artifacts", []).append(artifact)
    write_manifest(run_dir, manifest)
    append_jsonl(run_dir / "artifacts.jsonl", artifact)
    return artifact


def auto_collect_artifacts(project: Project, run_id: str) -> list[dict[str, Any]]:
    run_dir = project.runs_dir / run_id
    artifacts_dir = run_dir / "artifacts"
    collected: list[dict[str, Any]] = []
    if not artifacts_dir.exists():
        return collected
    existing_paths = {
        item.get("resolved_path") for item in load_manifest(run_dir).get("artifacts", [])
    }
    for path in sorted(item for item in artifacts_dir.rglob("*") if item.is_file()):
        if str(path.resolve()) in existing_paths:
            continue
        collected.append(
            register_artifact(
                project,
                run_id,
                path,
                claim_relevance="none",
                auto_collected=True,
            )
        )
    return collected


def annotate_artifact(
    project: Project,
    run_id: str,
    artifact_id: str,
    *,
    claim_relevance: str,
    description: str | None = None,
) -> dict[str, Any]:
    run_dir = project.runs_dir / run_id
    manifest = load_manifest(run_dir)
    for artifact in manifest.get("artifacts", []):
        if artifact["artifact_id"] == artifact_id:
            artifact["claim_relevance"] = claim_relevance
            artifact["review_status"] = "annotated" if claim_relevance != "none" else "ignored"
            if description is not None:
                artifact["description"] = description
            artifact["annotated_at"] = now_utc()
            write_manifest(run_dir, manifest)
            append_jsonl(run_dir / "artifacts.jsonl", {"event": "artifact_annotated", **artifact})
            return artifact
    raise KeyError(f"Artifact {artifact_id} not found in run {run_id}.")


def next_artifact_id(manifest: dict[str, Any]) -> str:
    return f"A{len(manifest.get('artifacts', [])) + 1:03d}"


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def resolve_artifact_path(project: Project, path: Path) -> Path:
    raw = str(path)
    if raw.startswith("//"):
        return (project.root / raw[2:]).resolve()
    if path.is_absolute():
        return path.resolve()
    return (project.root / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to_project(project: Project, path: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(project.root))
    except ValueError:
        return None


def _artifact_type(path: Path) -> str:
    if path.suffix:
        return path.suffix.lstrip(".")
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "file"
