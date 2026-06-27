from __future__ import annotations

import importlib
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import psutil
import torch


def bytes_to_gb(value: int | float) -> float:
    return float(value) / (1024**3)


def collect_resource_snapshot(path: str | Path = ".") -> dict[str, Any]:
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(path)
    cuda_available = torch.cuda.is_available()
    cuda: dict[str, Any] = {"available": cuda_available}
    if cuda_available:
        device_index = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        props = torch.cuda.get_device_properties(device_index)
        cuda.update(
            {
                "device_index": device_index,
                "gpu_name": props.name,
                "total_vram_gb": bytes_to_gb(total_bytes),
                "free_vram_gb": bytes_to_gb(free_bytes),
                "torch_device_total_memory_gb": bytes_to_gb(props.total_memory),
            }
        )
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda": cuda,
        "system_ram_total_gb": bytes_to_gb(memory.total),
        "system_ram_available_gb": bytes_to_gb(memory.available),
        "disk_total_gb": bytes_to_gb(disk.total),
        "disk_free_gb": bytes_to_gb(disk.free),
    }


def transformer_lens_import_status() -> dict[str, Any]:
    try:
        module = importlib.import_module("transformer_lens")
    except Exception as exc:  # pragma: no cover - exact dependency failure varies
        return {"ok": False, "error": repr(exc)}
    return {"ok": True, "version": getattr(module, "__version__", "unknown")}


def safe_initial_batch_size(
    max_examples: int,
    max_gpu_vram_fraction: float,
    free_vram_gb: float | None,
) -> int:
    if max_examples <= 0:
        return 1
    if free_vram_gb is None:
        return min(2, max_examples)
    usable_vram = free_vram_gb * max_gpu_vram_fraction
    if usable_vram >= 12:
        return min(16, max_examples)
    if usable_vram >= 8:
        return min(8, max_examples)
    if usable_vram >= 4:
        return min(4, max_examples)
    return 1
