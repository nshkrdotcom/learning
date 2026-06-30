from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def collect_attention_diagnostics(model: torch.nn.Module, step: int) -> list[dict[str, Any]]:
    rows = []
    blocks = getattr(getattr(model, "transformer", None), "h", [])
    for layer, block in enumerate(blocks):
        attention = getattr(block, "attn", None)
        diagnostics_fn = getattr(attention, "attention_diagnostics", None)
        if diagnostics_fn is None:
            continue
        row = diagnostics_fn(step=step, layer=layer)
        if row is not None:
            rows.append(row)
    return rows


def append_attention_diagnostics(out_dir: str | Path, rows: list[dict[str, Any]]) -> Path | None:
    if not rows:
        return None
    path = Path(out_dir) / "evals" / "attention_diagnostics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return path
