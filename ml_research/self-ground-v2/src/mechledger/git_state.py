from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

ENV_ALLOWLIST = {
    "PYTHONPATH",
    "CUDA_VISIBLE_DEVICES",
    "VIRTUAL_ENV",
    "CONDA_DEFAULT_ENV",
    "HF_HOME",
    "TRANSFORMERS_CACHE",
}
SECRET_MARKERS = (
    "TOKEN",
    "KEY",
    "SECRET",
    "PASSWORD",
    "AWS_SECRET",
    "OPENAI",
    "ANTHROPIC",
    "HF_TOKEN",
)


def git_metadata(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    commit = _git(root, ["rev-parse", "HEAD"])
    status = _git(root, ["status", "--short"])
    diff = _git(root, ["diff"])
    diff_hash = hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff is not None else None
    return {
        "git_commit": commit.strip() if commit else None,
        "git_dirty": bool(status.strip()) if status is not None else None,
        "git_status_short": status.splitlines() if status else [],
        "git_diff_hash": diff_hash,
    }


def captured_environment(env: dict[str, str] | None = None) -> dict[str, str]:
    env = env or dict(os.environ)
    captured: dict[str, str] = {}
    for key in sorted(ENV_ALLOWLIST):
        if key in env and not _is_secret(key):
            captured[key] = env[key]
    return captured


def _is_secret(key: str) -> bool:
    upper = key.upper()
    return any(marker in upper for marker in SECRET_MARKERS)


def _git(root: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout
