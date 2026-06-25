from __future__ import annotations

from pathlib import Path

from ruamel.yaml import YAML

HOOKS = [
    {
        "id": "mechledger-draft-check",
        "name": "mechledger draft check",
        "entry": "mechledger draft check --staged",
        "language": "system",
        "pass_filenames": False,
        "stages": ["pre-commit"],
    },
    {
        "id": "mechledger-index-check",
        "name": "mechledger index check",
        "entry": "mechledger index --check --staged",
        "language": "system",
        "pass_filenames": False,
        "stages": ["pre-commit"],
    },
]


def install_precommit_config(root: Path) -> Path:
    path = root / ".pre-commit-config.yaml"
    yaml = YAML()
    data = yaml.load(path.read_text(encoding="utf-8")) if path.exists() else None
    if not data:
        data = {"repos": []}
    repos = data.setdefault("repos", [])
    local = next((repo for repo in repos if repo.get("repo") == "local"), None)
    if local is None:
        local = {"repo": "local", "hooks": []}
        repos.append(local)
    hooks = local.setdefault("hooks", [])
    existing = {hook.get("id"): hook for hook in hooks}
    for hook in HOOKS:
        if hook["id"] in existing:
            existing[hook["id"]].update(hook)
        else:
            hooks.append(dict(hook))
    with path.open("w", encoding="utf-8") as handle:
        yaml.dump(data, handle)
    return path


def install_direct_hook(root: Path) -> Path:
    hooks_dir = root / ".git/hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    path = hooks_dir / "pre-commit"
    path.write_text(
        "#!/bin/sh\n"
        "mechledger draft check --staged || exit $?\n"
        "mechledger index --check --staged || exit $?\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path
