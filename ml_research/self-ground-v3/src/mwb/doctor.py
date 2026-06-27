from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mwb.events import read_events
from mwb.project import PROJECT_DIRS, Project, ProjectManager
from mwb.sqlite_index import SCHEMA_TABLES, existing_tables


@dataclass(frozen=True)
class DoctorReport:
    project: Project
    status: str
    errors: list[str]
    warnings: list[str]

    def render(self) -> str:
        lines = [
            f"project: {self.project.name}",
            f"root: {self.project.root}",
            f"workspace: {self.project.mechanism_dir.relative_to(self.project.root)}",
            f"database: {self.project.sqlite_path.relative_to(self.project.root)}",
            f"status: {self.status}",
        ]
        if self.errors:
            lines.append("errors:")
            lines.extend(f"- {error}" for error in self.errors)
        if self.warnings:
            lines.append("warnings:")
            lines.extend(f"- {warning}" for warning in self.warnings)
        return "\n".join(lines)


def run_doctor(root: Path | None = None) -> DoctorReport:
    project = ProjectManager.discover(root)
    errors: list[str] = []
    warnings: list[str] = []

    for relative in ["project.toml", "workbench.sqlite", "events.jsonl", *PROJECT_DIRS]:
        path = project.mechanism_dir / relative
        if not path.exists():
            errors.append(f"missing {path.relative_to(project.root)}")

    try:
        read_events(project.events_path)
    except ValueError as exc:
        errors.append(str(exc))

    if project.sqlite_path.exists():
        tables = existing_tables(project.sqlite_path)
        missing_tables = sorted(SCHEMA_TABLES - tables)
        if missing_tables:
            errors.append(f"missing sqlite tables: {', '.join(missing_tables)}")
    else:
        errors.append(f"missing {project.sqlite_path.relative_to(project.root)}")

    status = "ok" if not errors else "error"
    return DoctorReport(project=project, status=status, errors=errors, warnings=warnings)

