from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import yaml

from attention_lab.training.verify_run import RunVerificationError, load_jsonl_metrics


def _nonnull_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def summarize_run(run_dir: str | Path, *, write_json: bool = True) -> dict[str, Any]:
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise RunVerificationError(f"Missing metrics.jsonl: {metrics_path}")

    config_path = run_dir / "config.yaml"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    metrics = load_jsonl_metrics(metrics_path)
    train_rows = [row for row in metrics if row.get("event") == "train"]
    val_rows = [row for row in metrics if row.get("event") == "val"]
    checkpoint_rows = [row for row in metrics if row.get("event") == "checkpoint"]
    val_losses = [_nonnull_float(row.get("val_loss")) for row in val_rows]
    val_losses = [value for value in val_losses if value is not None]
    val_perplexities = [_nonnull_float(row.get("val_perplexity")) for row in val_rows]
    val_perplexities = [value for value in val_perplexities if value is not None]
    tokens_per_sec = [_nonnull_float(row.get("tokens_per_sec")) for row in train_rows]
    tokens_per_sec = [value for value in tokens_per_sec if value is not None]
    peak_vram_values = [_nonnull_float(row.get("peak_vram_mb")) for row in metrics]
    peak_vram_values = [value for value in peak_vram_values if value is not None]
    steps = [int(row["step"]) for row in metrics if row.get("step") is not None]

    summary = {
        "run_dir": str(run_dir),
        "run_name": config.get("run", {}).get("name", run_dir.name),
        "max_step": max(steps) if steps else None,
        "train_event_count": len(train_rows),
        "val_event_count": len(val_rows),
        "initial_val_loss": val_losses[0] if val_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "best_val_loss": min(val_losses) if val_losses else None,
        "initial_val_perplexity": val_perplexities[0] if val_perplexities else None,
        "final_val_perplexity": val_perplexities[-1] if val_perplexities else None,
        "median_tokens_per_sec": statistics.median(tokens_per_sec) if tokens_per_sec else None,
        "peak_vram_mb": max(peak_vram_values) if peak_vram_values else None,
        "checkpoint_count": len(checkpoint_rows),
    }

    if write_json:
        output_path = run_dir / "evals" / "run_summary.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    summary = summarize_run(args.run_dir)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

