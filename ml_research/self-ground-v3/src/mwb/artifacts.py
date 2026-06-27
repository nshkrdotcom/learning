from __future__ import annotations

import mimetypes
import shutil
from dataclasses import dataclass
from pathlib import Path

from mwb.hashing import sha256_file
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

    def to_dict(self) -> dict:
        return {
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
        }


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
        source = path.resolve()
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(source)

        artifact_root = self.project.mechanism_dir / "artifacts" / role
        artifact_root.mkdir(parents=True, exist_ok=True)
        final_path = source
        if copy_into_artifacts:
            final_path = artifact_root / source.name
            if source != final_path:
                shutil.copy2(source, final_path)

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
        )
        insert_payload(self.project.sqlite_path, "artifacts", record.artifact_ref, record.to_dict())
        return record

