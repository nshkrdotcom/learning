from __future__ import annotations

import hashlib
import os
from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    current = Path(start or os.getcwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / ".mechledger" / "project.json").exists():
            return candidate
    return current


def mechledger_dir(project_root: str | Path) -> Path:
    return Path(project_root) / ".mechledger"


def runs_dir(project_root: str | Path) -> Path:
    return mechledger_dir(project_root) / "runs"


def alias_cache_path(project_root: str | Path) -> Path:
    preferred = mechledger_dir(project_root) / "alias_cache.txt"
    try:
        preferred.parent.mkdir(parents=True, exist_ok=True)
        with preferred.parent.joinpath(".write_test").open("w", encoding="utf-8") as handle:
            handle.write("")
        preferred.parent.joinpath(".write_test").unlink(missing_ok=True)
        return preferred
    except OSError:
        digest = hashlib.sha256(str(Path(project_root).resolve()).encode("utf-8")).hexdigest()[:12]
        fallback = Path("/tmp") / f"mechledger_cache_{digest}" / "alias_cache.txt"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        return fallback


def index_path(project_root: str | Path) -> Path:
    preferred = mechledger_dir(project_root) / "index.sqlite"
    try:
        preferred.parent.mkdir(parents=True, exist_ok=True)
        with preferred.parent.joinpath(".write_test").open("w", encoding="utf-8") as handle:
            handle.write("")
        preferred.parent.joinpath(".write_test").unlink(missing_ok=True)
        return preferred
    except OSError:
        digest = hashlib.sha256(str(Path(project_root).resolve()).encode("utf-8")).hexdigest()[:12]
        fallback = Path("/tmp") / f"mechledger_cache_{digest}" / "index.sqlite"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        return fallback
