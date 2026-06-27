from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from local_mi_lab.paths import resolve_repo_path

DEFAULT_RESOURCES: dict[str, Any] = {
    "max_disk_cache_gb": 100,
    "max_activation_cache_gb_per_run": 20,
    "max_ram_gb_per_run": 64,
    "max_gpu_vram_fraction": 0.80,
    "max_examples_initial": 128,
    "max_examples_full": 1024,
    "activation_cache_dtype": "float16",
    "batch_size_auto": True,
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_repo_path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    config = _with_resource_defaults(config)
    _validate_minimal_config(config, config_path)
    return config


def write_config(config: dict[str, Any], path: str | Path) -> None:
    output_path = resolve_repo_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def config_to_yaml(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False)


def _with_resource_defaults(config: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(config)
    resources = dict(DEFAULT_RESOURCES)
    resources.update(merged.get("resources", {}) or {})
    merged["resources"] = resources
    return merged


def _validate_minimal_config(config: dict[str, Any], path: Path) -> None:
    for key in ["experiment", "model", "resources", "task", "outputs"]:
        if key not in config:
            raise ValueError(f"Missing required config section {key!r} in {path}")
    if config["model"].get("backend") != "transformer_lens":
        raise ValueError("Only transformer_lens backend is supported in the first pass")
    if not config["model"].get("name"):
        raise ValueError("model.name is required")
    if not config["experiment"].get("name"):
        raise ValueError("experiment.name is required")


def experiment_name(config: dict[str, Any]) -> str:
    return str(config["experiment"]["name"])


def model_name(config: dict[str, Any]) -> str:
    return str(config["model"]["name"])


def run_root(config: dict[str, Any]) -> str:
    return str(config.get("outputs", {}).get("run_root", "runs"))


def resource_limit(config: dict[str, Any], key: str) -> Any:
    return config["resources"][key]


def evenly_spaced_layers(n_layers: int, count: int) -> list[int]:
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    if count >= n_layers:
        return list(range(n_layers))
    return sorted({round(i * (n_layers - 1) / (count - 1)) for i in range(count)})


def selected_layers(config: dict[str, Any], n_layers: int) -> list[int]:
    spec = (config.get("activations") or {}).get("layers", "auto_even_6")
    if spec == "auto_even_6":
        return evenly_spaced_layers(n_layers, 6)
    if spec == "all":
        return list(range(n_layers))
    if isinstance(spec, list) and all(isinstance(layer, int) for layer in spec):
        bad_layers = [layer for layer in spec if layer < 0 or layer >= n_layers]
        if bad_layers:
            raise ValueError(f"Layer indices out of range for {n_layers} layers: {bad_layers}")
        return spec
    raise ValueError(f"Unsupported activation layer spec: {spec!r}")


def max_examples_for_initial_run(config: dict[str, Any]) -> int:
    task_n = int(config.get("task", {}).get("n_examples_initial", 0) or 0)
    budget_n = int(config["resources"]["max_examples_initial"])
    if task_n <= 0:
        return budget_n
    return min(task_n, budget_n)
