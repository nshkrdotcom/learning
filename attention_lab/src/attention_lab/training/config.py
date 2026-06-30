from pathlib import Path
from typing import Any

import yaml

from attention_lab.models.gpt import GPTConfig


REQUIRED_SECTIONS = ("run", "data", "model", "train")
OPTIONAL_SECTIONS = ("sample", "status")
IMPLEMENTED_ATTENTION_TYPES = {"standard"}
KNOWN_ATTENTION_TYPES = {"standard", "cp_bilinear", "trilinear_cp"}
DTYPES = {"bfloat16", "float16", "float32"}
EXPERIMENTAL_UNIMPLEMENTED_STATUS = "experimental_unimplemented"
RUN_KEYS = {"name", "out_dir", "seed"}
DATA_KEYS = {"data_root", "tokenizer", "vocab_size", "train_tokens", "val_tokens", "dataset", "dataset_config"}
TRAIN_KEYS = {
    "device",
    "dtype",
    "compile",
    "eval_at_start",
    "B",
    "T",
    "total_batch_size",
    "max_steps",
    "grad_clip",
    "weight_decay",
    "learning_rate",
    "min_lr",
    "warmup_steps",
    "val_every",
    "val_steps",
    "save_every",
    "log_every",
}
SAMPLE_KEYS = {
    "sample_every",
    "prompt",
    "num_samples",
    "max_new_tokens",
    "top_k",
    "temperature",
    "seed",
}


def _require_mapping(config: dict[str, Any], section: str) -> dict[str, Any]:
    value = config.get(section)
    if not isinstance(value, dict):
        raise ValueError(f"{section} must be a mapping")
    return value


def _require_nonempty_string(section: dict[str, Any], key: str, dotted_key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{dotted_key} must be a nonempty string")
    return value


def _require_positive_int(section: dict[str, Any], key: str, dotted_key: str) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{dotted_key} must be a positive integer")
    return value


def _reject_unknown_keys(section: dict[str, Any], allowed_keys: set[str], section_name: str) -> None:
    unknown = sorted(set(section) - allowed_keys)
    if unknown:
        raise ValueError(f"Unknown {section_name} config keys: {unknown}")


def validate_config(
    config: dict[str, Any],
    *,
    source: str | Path | None = None,
    allow_experimental: bool = False,
) -> dict[str, Any]:
    source_text = f" {source}" if source is not None else ""
    if not isinstance(config, dict):
        raise ValueError(f"Config{source_text} must be a mapping")

    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"Config{source_text} is missing sections: {missing}")

    allowed_top_level = set(REQUIRED_SECTIONS) | set(OPTIONAL_SECTIONS)
    unknown_top_level = sorted(set(config) - allowed_top_level)
    if unknown_top_level:
        raise ValueError(f"Config{source_text} has unknown top-level sections: {unknown_top_level}")

    status = config.get("status")
    if status == EXPERIMENTAL_UNIMPLEMENTED_STATUS and not allow_experimental:
        raise ValueError(
            f"Config{source_text} is experimental/unimplemented and is not runnable baseline configuration"
        )

    run = _require_mapping(config, "run")
    data = _require_mapping(config, "data")
    model = _require_mapping(config, "model")
    train = _require_mapping(config, "train")
    sample = config.get("sample", {})
    if not isinstance(sample, dict):
        raise ValueError("sample must be a mapping")

    _reject_unknown_keys(run, RUN_KEYS, "run")
    _reject_unknown_keys(data, DATA_KEYS, "data")
    _reject_unknown_keys(train, TRAIN_KEYS, "train")
    _reject_unknown_keys(sample, SAMPLE_KEYS, "sample")

    _require_nonempty_string(run, "name", "run.name")
    _require_nonempty_string(run, "out_dir", "run.out_dir")
    _require_nonempty_string(data, "data_root", "data.data_root")
    _require_positive_int(data, "vocab_size", "data.vocab_size")

    valid_model_keys = set(GPTConfig.__dataclass_fields__)
    unknown_model_keys = sorted(set(model) - valid_model_keys)
    if unknown_model_keys:
        raise ValueError(f"Unknown model config keys: {unknown_model_keys}")

    attention_type = model.get("attention_type", "standard")
    if attention_type not in KNOWN_ATTENTION_TYPES:
        raise ValueError(f"model.attention_type must be one of {sorted(KNOWN_ATTENTION_TYPES)}")
    if attention_type not in IMPLEMENTED_ATTENTION_TYPES and not allow_experimental:
        raise ValueError(f"model.attention_type={attention_type!r} is not implemented for baseline runs")

    block_size = _require_positive_int(model, "block_size", "model.block_size")
    n_layer = _require_positive_int(model, "n_layer", "model.n_layer")
    n_head = _require_positive_int(model, "n_head", "model.n_head")
    n_embd = _require_positive_int(model, "n_embd", "model.n_embd")
    if n_layer <= 0 or block_size <= 0:
        raise ValueError("model dimensions must be positive")
    if n_embd % n_head != 0:
        raise ValueError("model.n_embd must be divisible by model.n_head")

    B = _require_positive_int(train, "B", "train.B")
    T = _require_positive_int(train, "T", "train.T")
    total_batch_size = _require_positive_int(train, "total_batch_size", "train.total_batch_size")
    if total_batch_size % (B * T) != 0:
        raise ValueError("train.total_batch_size must be divisible by train.B * train.T for single-process runs")

    for key in ("max_steps", "val_every", "val_steps", "save_every", "log_every"):
        _require_positive_int(train, key, f"train.{key}")

    dtype = train.get("dtype")
    if dtype not in DTYPES:
        raise ValueError(f"train.dtype must be one of {sorted(DTYPES)}")

    if bool(train.get("compile", False)):
        raise ValueError("train.compile is not supported for baseline QC; set compile: false")

    return config


def load_config(path: str | Path, *, allow_experimental: bool = False) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return validate_config(config, source=path, allow_experimental=allow_experimental)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
