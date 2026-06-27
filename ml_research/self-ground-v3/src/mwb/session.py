from __future__ import annotations

import json
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mwb.domain.objects import WorkbenchObject
from mwb.events import append_event
from mwb.git_state import capture_git_state
from mwb.hashing import sha256_text
from mwb.project import Project
from mwb.refs import stable_ref
from mwb.sqlite_index import insert_event, insert_payload
from mwb.time import utc_now


@dataclass
class CellState:
    ref: str
    source: str
    execution_index: int
    namespace_before: dict[str, Any]
    started_at: str
    status: str = "running"
    ended_at: str | None = None
    created_object_refs: list[str] = field(default_factory=list)
    mutated_object_refs: list[str] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)
    namespace_events: list[dict[str, Any]] = field(default_factory=list)
    stdout_ref: str | None = None
    stderr_ref: str | None = None
    exception_ref: str | None = None


class WorkbenchSession:
    def __init__(
        self,
        *,
        project: Project,
        session_ref: str,
        session_dir: Path,
        surface: str,
        mode: str = "scratch",
        resumed_from_session_ref: str | None = None,
    ) -> None:
        self.project = project
        self.session_ref = session_ref
        self.session_dir = session_dir
        self.surface = surface
        self.mode = mode
        self.resumed_from_session_ref = resumed_from_session_ref
        self.started_at = utc_now()
        self.ended_at: str | None = None
        self.current_cell: CellState | None = None
        self._cell_counter = self._initial_cell_counter()
        self.git_state = capture_git_state(project.root)
        self.environment = {
            "pid": os.getpid(),
            "python_executable": os.sys.executable,
            "cwd": str(Path.cwd()),
        }
        self._ensure_layout()
        self.write_session_json()
        insert_payload(
            self.project.sqlite_path,
            "sessions",
            self.session_ref,
            self.session_payload(),
        )
        event = append_event(
            self.project.events_path,
            "session_started",
            {"session_ref": self.session_ref, "surface": self.surface, "mode": self.mode},
        )
        insert_event(self.project.sqlite_path, event)

    def _ensure_layout(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        for directory in ["stdout", "stderr", "exceptions", "snapshots"]:
            (self.session_dir / directory).mkdir(parents=True, exist_ok=True)
        for file_name in ["cells.jsonl", "namespace_objects.jsonl", "artifacts.jsonl"]:
            (self.session_dir / file_name).touch(exist_ok=True)

    def _initial_cell_counter(self) -> int:
        cells_path = self.session_dir / "cells.jsonl"
        if not cells_path.exists():
            return 0
        return len([line for line in cells_path.read_text().splitlines() if line.strip()])

    def session_payload(self) -> dict[str, Any]:
        return {
            "session_ref": self.session_ref,
            "project_ref": self.project.project_ref,
            "surface": self.surface,
            "mode": self.mode,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "workspace": str(self.project.mechanism_dir.relative_to(self.project.root)),
            "sqlite_path": str(self.project.sqlite_path.relative_to(self.project.root)),
            "git_state": self.git_state,
            "environment": self.environment,
            "operator_ref": "local_user",
            "extension_version": "0.1.0",
            "resumed_from_session_ref": self.resumed_from_session_ref,
        }

    def write_session_json(self) -> None:
        (self.session_dir / "session.json").write_text(
            json.dumps(self.session_payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def begin_cell(
        self, *, source: str, execution_index: int, namespace_before: dict[str, Any]
    ) -> CellState:
        self._cell_counter += 1
        cell_ref = f"cell_{self._cell_counter:06d}"
        state = CellState(
            ref=cell_ref,
            source=source,
            execution_index=execution_index,
            namespace_before=namespace_before,
            started_at=utc_now(),
        )
        self.current_cell = state
        source_path = self.session_dir / "snapshots" / f"{cell_ref}_source.py"
        source_path.write_text(source, encoding="utf-8")
        return state

    def finish_cell_execution(self, *, status: str, error: BaseException | None = None) -> None:
        if self.current_cell is None:
            return
        self.current_cell.status = status
        self.current_cell.ended_at = utc_now()
        if error is not None:
            exception_path = self.session_dir / "exceptions" / f"{self.current_cell.ref}.txt"
            exception_path.write_text(
                "".join(traceback.format_exception(type(error), error, error.__traceback__)),
                encoding="utf-8",
            )
            self.current_cell.exception_ref = stable_ref(
                "art", self.session_ref, self.current_cell.ref, "exception"
            )

    def record_cell_streams(
        self,
        cell: CellState,
        *,
        stdout_text: str,
        stderr_text: str,
        max_bytes: int = 65536,
    ) -> None:
        if stdout_text:
            stored = _bounded_text(stdout_text, max_bytes=max_bytes)
            (self.session_dir / "stdout" / f"{cell.ref}.txt").write_text(
                stored,
                encoding="utf-8",
            )
            cell.stdout_ref = stable_ref("art", self.session_ref, cell.ref, "stdout", stored)
        if stderr_text:
            stored = _bounded_text(stderr_text, max_bytes=max_bytes)
            (self.session_dir / "stderr" / f"{cell.ref}.txt").write_text(
                stored,
                encoding="utf-8",
            )
            cell.stderr_ref = stable_ref("art", self.session_ref, cell.ref, "stderr", stored)

    def register_object_event(
        self,
        *,
        event_type: str,
        variable_name: str,
        obj: WorkbenchObject,
        cell: CellState,
    ) -> None:
        record = {
            "event": event_type,
            "session_ref": self.session_ref,
            "cell_ref": cell.ref,
            "variable_name": variable_name,
            "object_ref": obj.wb_ref,
            "object_type": obj.wb_type,
            "workbench_ref": obj.wb_ref,
            "content_hash": obj.wb_fingerprint(),
            "parents": obj.wb_parents(),
            "object_payload": obj.model_dump(mode="json"),
            "created_at": utc_now(),
        }
        self._append_jsonl("namespace_objects.jsonl", record)
        cell.namespace_events.append(record)
        if event_type == "object_registered":
            cell.created_object_refs.append(obj.wb_ref)
            insert_payload(
                self.project.sqlite_path,
                "objects",
                obj.wb_ref,
                obj.model_dump(mode="json"),
            )
            self._insert_lineage_edges(obj=obj, cell=cell, relation="created_in_cell")
        elif event_type == "object_mutated":
            cell.mutated_object_refs.append(obj.wb_ref)
            insert_payload(
                self.project.sqlite_path,
                "object_versions",
                stable_ref("objver", obj.wb_ref, obj.wb_version, obj.wb_fingerprint()),
                obj.model_dump(mode="json"),
            )
            self._insert_lineage_edges(obj=obj, cell=cell, relation="mutated_in_cell")
        event = append_event(
            self.project.events_path,
            event_type,
            {
                "session_ref": self.session_ref,
                "cell_ref": cell.ref,
                "variable_name": variable_name,
                "object_ref": obj.wb_ref,
            },
        )
        insert_event(self.project.sqlite_path, event)

    def commit_cell(self, cell: CellState) -> None:
        source_hash = sha256_text(cell.source)
        source_artifact_ref = stable_ref("art", self.session_ref, cell.ref, source_hash)
        record = {
            "cell_ref": cell.ref,
            "session_ref": self.session_ref,
            "execution_index": cell.execution_index,
            "source_hash": source_hash,
            "source_text_artifact_ref": source_artifact_ref,
            "started_at": cell.started_at,
            "ended_at": cell.ended_at,
            "status": cell.status,
            "created_object_refs": cell.created_object_refs,
            "mutated_object_refs": cell.mutated_object_refs,
            "artifact_refs": cell.artifact_refs,
            "stdout_ref": cell.stdout_ref,
            "stderr_ref": cell.stderr_ref,
            "exception_ref": cell.exception_ref,
        }
        self._append_jsonl("cells.jsonl", record)
        insert_payload(self.project.sqlite_path, "cells", cell.ref, record)
        event = append_event(
            self.project.events_path,
            "cell_executed",
            {"session_ref": self.session_ref, "cell_ref": cell.ref, "status": cell.status},
        )
        insert_event(self.project.sqlite_path, event)
        self.current_cell = None

    def close(self) -> None:
        self.ended_at = utc_now()
        self.write_session_json()
        insert_payload(
            self.project.sqlite_path,
            "sessions",
            self.session_ref,
            self.session_payload(),
        )

    def _append_jsonl(self, file_name: str, record: dict[str, Any]) -> None:
        with (self.session_dir / file_name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _insert_lineage_edges(
        self,
        *,
        obj: WorkbenchObject,
        cell: CellState,
        relation: str,
    ) -> None:
        created_at = utc_now()
        cell_edge = {
            "src_ref": cell.ref,
            "dst_ref": obj.wb_ref,
            "relation": relation,
            "session_ref": self.session_ref,
            "cell_ref": cell.ref,
            "created_at": created_at,
        }
        insert_payload(
            self.project.sqlite_path,
            "lineage_edges",
            stable_ref("edge", cell.ref, obj.wb_ref, relation),
            cell_edge,
        )
        for parent_ref in obj.wb_parents():
            parent_edge = {
                "src_ref": parent_ref,
                "dst_ref": obj.wb_ref,
                "relation": "parent",
                "session_ref": self.session_ref,
                "cell_ref": cell.ref,
                "created_at": created_at,
            }
            insert_payload(
                self.project.sqlite_path,
                "lineage_edges",
                stable_ref("edge", parent_ref, obj.wb_ref, "parent"),
                parent_edge,
            )


def _bounded_text(value: str, *, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    suffix = "\n[truncated by mwb capture]\n"
    suffix_bytes = suffix.encode("utf-8")
    trimmed = encoded[: max(0, max_bytes - len(suffix_bytes))]
    return trimmed.decode("utf-8", errors="ignore") + suffix


class SessionManager:
    @staticmethod
    def start(
        project: Project,
        *,
        surface: str,
        resume: str | None = None,
        mode: str = "scratch",
    ) -> WorkbenchSession:
        if resume:
            session_ref = resume
            session_dir = project.mechanism_dir / "sessions" / session_ref
            if not session_dir.exists():
                raise FileNotFoundError(f"cannot resume missing session {session_ref}")
            new_ref = stable_ref("sess", resume, utc_now(), os.getpid())
            return WorkbenchSession(
                project=project,
                session_ref=new_ref,
                session_dir=project.mechanism_dir / "sessions" / new_ref,
                surface=surface,
                mode=mode,
                resumed_from_session_ref=resume,
            )
        session_ref = stable_ref("sess", utc_now(), os.getpid(), len(list_sessions(project)))
        return WorkbenchSession(
            project=project,
            session_ref=session_ref,
            session_dir=project.mechanism_dir / "sessions" / session_ref,
            surface=surface,
            mode=mode,
        )


def list_sessions(project: Project) -> list[Path]:
    sessions_dir = project.mechanism_dir / "sessions"
    if not sessions_dir.exists():
        return []
    return sorted(path for path in sessions_dir.glob("sess_*") if path.is_dir())


def latest_session(project: Project) -> Path:
    sessions = list_sessions(project)
    if not sessions:
        raise FileNotFoundError("no sessions found")
    return sessions[-1]
