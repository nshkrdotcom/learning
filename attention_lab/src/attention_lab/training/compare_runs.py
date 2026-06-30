from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COMPARISON_FIELDS = [
    "run_name",
    "final_val_loss",
    "best_val_loss",
    "final_val_perplexity",
    "median_tokens_per_sec",
    "peak_vram_allocated_mb",
    "peak_vram_mb",
    "peak_vram_reserved_mb",
    "checkpoint_count",
]


def load_run_summary(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    summary_path = run_dir / "evals" / "run_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing run summary: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    if "peak_vram_allocated_mb" not in summary:
        summary["peak_vram_allocated_mb"] = summary.get("peak_vram_mb")
    return summary


def compare_runs(baseline: str | Path, candidate: str | Path) -> list[dict[str, Any]]:
    return [
        {"role": "baseline", **load_run_summary(baseline)},
        {"role": "candidate", **load_run_summary(candidate)},
    ]


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_comparison(rows: list[dict[str, Any]]) -> None:
    fields = ["role", *COMPARISON_FIELDS]
    widths = {
        field: max(len(field), *(len(_format_value(row.get(field))) for row in rows))
        for field in fields
    }
    print("  ".join(field.ljust(widths[field]) for field in fields))
    print("  ".join("-" * widths[field] for field in fields))
    for row in rows:
        print("  ".join(_format_value(row.get(field)).ljust(widths[field]) for field in fields))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    rows = compare_runs(args.baseline, args.candidate)
    print_comparison(rows)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
