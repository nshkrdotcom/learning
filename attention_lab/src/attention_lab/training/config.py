from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = ("run", "data", "model", "train")


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"Config {path} is missing sections: {missing}")
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

