from __future__ import annotations

from pathlib import Path

PRE_COMMIT_SNIPPET = """\
repos:
  - repo: local
    hooks:
      - id: mechledger-draft-check
        name: mechledger draft check
        entry: mechledger draft check --staged
        language: system
        pass_filenames: false
        stages: [pre-commit]
      - id: mechledger-index-check
        name: mechledger index check
        entry: mechledger index --check --staged
        language: system
        pass_filenames: false
        stages: [pre-commit]
"""


def install_pre_commit_config(project_root: str | Path) -> Path:
    path = Path(project_root) / ".pre-commit-config.yaml"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if "mechledger-draft-check" in text:
            return path
        if text and not text.endswith("\n"):
            text += "\n"
        text += "\n" + PRE_COMMIT_SNIPPET
    else:
        text = PRE_COMMIT_SNIPPET
    path.write_text(text, encoding="utf-8")
    return path


def install_direct_hook(project_root: str | Path) -> Path:
    hook = Path(project_root) / ".git" / "hooks" / "pre-commit"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(
        "#!/usr/bin/env sh\n"
        "mechledger draft check --staged || exit $?\n"
        "mechledger index --check --staged || exit $?\n",
        encoding="utf-8",
    )
    hook.chmod(0o755)
    return hook
