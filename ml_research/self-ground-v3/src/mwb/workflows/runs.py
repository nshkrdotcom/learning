from __future__ import annotations

from pathlib import Path

from mwb.project import Project, ProjectManager


def resolve_run_path(value: Path, *, project: Project | None = None) -> Path:
    """Resolve a run directory argument, including the local `latest` alias."""
    if str(value) == "latest":
        return latest_run_path(project=project)
    if value.exists():
        return value
    if value.is_absolute():
        return value

    project = project or ProjectManager.discover()
    candidate = project.mechanism_dir / "runs" / str(value)
    if candidate.exists():
        return candidate
    return value


def latest_run_path(*, project: Project | None = None) -> Path:
    project = project or ProjectManager.discover()
    runs_dir = project.mechanism_dir / "runs"
    candidates = [
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and (path / "run_manifest.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"no runs found in {runs_dir}")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))
