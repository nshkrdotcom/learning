from __future__ import annotations

from datetime import datetime
from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    for path in [current, *current.parents]:
        if (path / "pyproject.toml").exists():
            return path
    raise RuntimeError(f"Could not find repository root from {current}")


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(run_root: str | Path, suffix: str) -> Path:
    root = resolve_repo_path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"{timestamp()}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def relative_files(root: Path) -> list[str]:
    files: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(path.relative_to(root).as_posix())
    return files
