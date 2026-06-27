from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from mwb.hashing import sha256_text
from mwb.time import utc_now


def _git(root: Path, *args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=check,
        capture_output=True,
        text=True,
    )


def discover_git_root(start: Path) -> Path | None:
    result = _git(start, "rev-parse", "--show-toplevel")
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()).resolve()


def capture_git_state(root: Path) -> dict[str, Any]:
    repo_root = discover_git_root(root) or root.resolve()
    branch_result = _git(repo_root, "branch", "--show-current")
    commit_result = _git(repo_root, "rev-parse", "HEAD")
    status_result = _git(repo_root, "status", "--porcelain=v1")
    diff_result = _git(repo_root, "diff", "--binary")

    status = status_result.stdout
    untracked = [line[3:] for line in status.splitlines() if line.startswith("?? ")]
    dirty = bool(status.strip())
    diff_material = status + "\n" + diff_result.stdout

    return {
        "repo_root": str(repo_root),
        "branch": branch_result.stdout.strip() if branch_result.returncode == 0 else None,
        "commit": commit_result.stdout.strip() if commit_result.returncode == 0 else None,
        "dirty": dirty,
        "diff_hash": sha256_text(diff_material),
        "untracked": untracked,
        "timestamp": utc_now(),
    }

