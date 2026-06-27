from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from local_mi_lab.config import selected_layers
from local_mi_lab.models import n_layers
from local_mi_lab.types import PromptRecord


def resid_post_hook_name(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_post"


def selected_resid_post_hooks(config: dict[str, Any], model: Any) -> list[str]:
    return [resid_post_hook_name(layer) for layer in selected_layers(config, n_layers(model))]


def resolve_position_index(position: str | int, seq_len: int) -> int:
    if isinstance(position, int):
        index = position
    elif position == "final":
        index = seq_len - 1
    else:
        raise ValueError(f"Unsupported token position {position!r}")
    if index < 0:
        index = seq_len + index
    if index < 0 or index >= seq_len:
        raise ValueError(f"Position {position!r} out of range for sequence length {seq_len}")
    return index


def cache_selected_resid_post(
    model: Any,
    records: list[PromptRecord],
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    layers = selected_layers(config, n_layers(model))
    hook_names = [resid_post_hook_name(layer) for layer in layers]
    positions = (config.get("activations") or {}).get("token_positions", ["final"])
    if positions != ["final"]:
        raise ValueError("First-pass activation caching only supports token_positions: [final]")

    activation_dir = output_dir / "activations"
    activation_dir.mkdir(parents=True, exist_ok=True)
    dtype_name = str(config["resources"].get("activation_cache_dtype", "float16"))
    cache_dtype = torch.float16 if dtype_name == "float16" else torch.float32

    layer_tensors: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    for record in tqdm(records, desc="Caching selected activations"):
        tokens = model.to_tokens(record.prompt)
        with torch.inference_mode():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)
        final_index = resolve_position_index("final", tokens.shape[1])
        for layer in layers:
            tensor = cache[resid_post_hook_name(layer)][0, final_index, :].detach().to("cpu", cache_dtype)
            layer_tensors[layer].append(tensor)

    files: list[dict[str, Any]] = []
    for layer, tensors in layer_tensors.items():
        stacked = torch.stack(tensors, dim=0)
        file_name = f"layer_{layer:02d}_resid_post.pt"
        file_path = activation_dir / file_name
        torch.save(stacked, file_path)
        files.append(
            {
                "layer": layer,
                "hook_name": resid_post_hook_name(layer),
                "path": f"activations/{file_name}",
                "shape": list(stacked.shape),
                "dtype": str(stacked.dtype),
                "file_size_bytes": file_path.stat().st_size,
            }
        )

    manifest = {
        "model": config["model"]["name"],
        "task": config["task"]["name"],
        "hook_names": hook_names,
        "layers": layers,
        "token_positions": positions,
        "dtype": dtype_name,
        "n_examples": len(records),
        "families_present": sorted({record.family for record in records}),
        "n_examples_by_family": _counts_by_family(records),
        "files": files,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cache_budget": {
            "max_activation_cache_gb_per_run": config["resources"][
                "max_activation_cache_gb_per_run"
            ],
            "cache_strategy": (config.get("activations") or {}).get("cache_strategy"),
        },
    }
    (activation_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _counts_by_family(records: list[PromptRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.family] = counts.get(record.family, 0) + 1
    return dict(sorted(counts.items()))
