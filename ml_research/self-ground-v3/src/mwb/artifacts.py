from __future__ import annotations

import mimetypes
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from mwb.hashing import sha256_file, sha256_text
from mwb.project import Project
from mwb.refs import stable_ref
from mwb.sqlite_index import insert_payload
from mwb.time import utc_now


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_ref: str
    path: str
    role: str
    sha256: str
    byte_count: int
    mime_type: str
    created_at: str
    created_by_ref: str | None
    parents: list[str]
    redaction_posture: str = "local"
    materialized: bool = True
    pointer: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        payload = {
            "artifact_ref": self.artifact_ref,
            "path": self.path,
            "role": self.role,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "mime_type": self.mime_type,
            "created_at": self.created_at,
            "created_by_ref": self.created_by_ref,
            "parents": self.parents,
            "redaction_posture": self.redaction_posture,
            "materialized": self.materialized,
        }
        if self.pointer is not None:
            payload["pointer"] = self.pointer
        return payload


class ArtifactRegistry:
    def __init__(self, project: Project) -> None:
        self.project = project

    def register_path(
        self,
        path: Path,
        *,
        role: str,
        created_by_ref: str | None = None,
        parents: list[str] | None = None,
        copy_into_artifacts: bool = False,
    ) -> ArtifactRecord:
        requested = path if path.is_absolute() else self.project.root / path
        symlink_pointer = _git_annex_pointer_from_symlink(requested)
        if symlink_pointer is not None:
            record = self._record_pointer(
                requested,
                role=role,
                pointer=symlink_pointer,
                created_by_ref=created_by_ref,
                parents=parents,
            )
            insert_payload(
                self.project.sqlite_path,
                "artifacts",
                record.artifact_ref,
                record.to_dict(),
            )
            return record

        source = requested.resolve()
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(source)

        artifact_root = self.project.mechanism_dir / "artifacts" / role
        artifact_root.mkdir(parents=True, exist_ok=True)
        final_path = source
        if copy_into_artifacts:
            final_path = artifact_root / source.name
            if source != final_path:
                shutil.copy2(source, final_path)

        pointer = detect_artifact_pointer(final_path)
        artifact_hash = sha256_file(final_path)
        byte_count = final_path.stat().st_size
        try:
            relative = final_path.relative_to(self.project.root)
            stored_path = str(relative)
        except ValueError:
            stored_path = str(final_path)
        mime_type = mimetypes.guess_type(final_path.name)[0] or "application/octet-stream"
        artifact_ref = stable_ref("art", stored_path, role, artifact_hash)
        record = ArtifactRecord(
            artifact_ref=artifact_ref,
            path=stored_path,
            role=role,
            sha256=artifact_hash,
            byte_count=byte_count,
            mime_type=mime_type,
            created_at=utc_now(),
            created_by_ref=created_by_ref,
            parents=list(parents or []),
            materialized=pointer is None,
            pointer=pointer,
        )
        insert_payload(self.project.sqlite_path, "artifacts", record.artifact_ref, record.to_dict())
        return record

    def _record_pointer(
        self,
        path: Path,
        *,
        role: str,
        pointer: dict[str, Any],
        created_by_ref: str | None,
        parents: list[str] | None,
    ) -> ArtifactRecord:
        target = str(pointer.get("target", ""))
        artifact_hash = sha256_text(target)
        try:
            stored_path = str(path.relative_to(self.project.root))
        except ValueError:
            stored_path = str(path)
        return ArtifactRecord(
            artifact_ref=stable_ref("art", stored_path, role, artifact_hash),
            path=stored_path,
            role=role,
            sha256=artifact_hash,
            byte_count=len(target.encode("utf-8")),
            mime_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            created_at=utc_now(),
            created_by_ref=created_by_ref,
            parents=list(parents or []),
            materialized=False,
            pointer=pointer,
        )


def detect_artifact_pointer(path: Path) -> dict[str, Any] | None:
    text = _read_small_text(path)
    if text is None:
        return None
    git_lfs = _git_lfs_pointer(text)
    if git_lfs is not None:
        return git_lfs
    if path.name.endswith(".dvc"):
        return _dvc_pointer(text)
    return None


def _read_small_text(path: Path, *, max_bytes: int = 128 * 1024) -> str | None:
    try:
        if path.stat().st_size > max_bytes:
            return None
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def _git_lfs_pointer(text: str) -> dict[str, Any] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or lines[0] != "version https://git-lfs.github.com/spec/v1":
        return None
    fields: dict[str, str] = {}
    for line in lines[1:]:
        if " " not in line:
            continue
        key, value = line.split(" ", 1)
        fields[key] = value
    oid = fields.get("oid")
    size = fields.get("size")
    if oid is None or size is None:
        return None
    return {"backend": "git_lfs", "oid": oid, "size": int(size)}


def _dvc_pointer(text: str) -> dict[str, Any] | None:
    yaml = YAML(typ="safe")
    payload = yaml.load(text) or {}
    outs = payload.get("outs")
    if not isinstance(outs, list) or not outs:
        return None
    first = outs[0]
    if not isinstance(first, dict):
        return None
    path = first.get("path")
    oid = first.get("md5") or first.get("etag") or first.get("checksum")
    result: dict[str, Any] = {"backend": "dvc"}
    if path is not None:
        result["path"] = str(path)
    if oid is not None:
        oid_name = "md5" if first.get("md5") else "oid"
        result["oid"] = f"{oid_name}:{oid}"
    if first.get("size") is not None:
        result["size"] = int(first["size"])
    return result


def _git_annex_pointer_from_symlink(path: Path) -> dict[str, Any] | None:
    if not path.is_symlink():
        return None
    target = os.readlink(path)
    key = Path(target).name
    if "annex/objects" not in target and "--" not in key:
        return None
    result: dict[str, Any] = {
        "backend": "git_annex",
        "target": target,
        "key": key,
    }
    match = re.search(r"-s(\d+)--", key)
    if match:
        result["size"] = int(match.group(1))
    return result
