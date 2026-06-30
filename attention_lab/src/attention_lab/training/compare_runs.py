from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from attention_lab.training.experiments import get_experiment


COMPARISON_FIELDS = [
    "run_name",
    "final_val_loss",
    "best_val_loss",
    "final_val_perplexity",
    "median_tokens_per_sec",
    "peak_vram_allocated_mb",
    "peak_vram_mb",
    "peak_vram_reserved_mb",
    "nvidia_smi_memory_mb",
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


def _numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(candidate: dict[str, Any], baseline: dict[str, Any], field: str) -> float | None:
    candidate_value = _numeric(candidate.get(field))
    baseline_value = _numeric(baseline.get(field))
    if candidate_value is None or baseline_value is None:
        return None
    return candidate_value - baseline_value


def _ratio(candidate: dict[str, Any], baseline: dict[str, Any], field: str) -> float | None:
    candidate_value = _numeric(candidate.get(field))
    baseline_value = _numeric(baseline.get(field))
    if candidate_value is None or baseline_value in {None, 0.0}:
        return None
    return candidate_value / baseline_value


def derived_metrics(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float | None]:
    return {
        "delta_final_val_loss": _delta(candidate, baseline, "final_val_loss"),
        "delta_best_val_loss": _delta(candidate, baseline, "best_val_loss"),
        "delta_final_val_perplexity": _delta(candidate, baseline, "final_val_perplexity"),
        "tokens_per_sec_ratio": _ratio(candidate, baseline, "median_tokens_per_sec"),
        "peak_vram_allocated_ratio": _ratio(candidate, baseline, "peak_vram_allocated_mb"),
        "peak_vram_reserved_ratio": _ratio(candidate, baseline, "peak_vram_reserved_mb"),
    }


def compare_runs_for_experiment(
    experiment_id: str,
    baseline: str | Path,
    candidate: str | Path,
) -> dict[str, Any]:
    experiment = get_experiment(experiment_id)
    candidate_path = Path(candidate).resolve()
    experiment_run_dir = Path(experiment["run_dir"]).resolve()
    try:
        candidate_path.relative_to(experiment_run_dir)
    except ValueError as exc:
        raise ValueError(
            f"Candidate run {candidate_path} is not under experiment run dir {experiment_run_dir}"
        ) from exc
    rows = compare_runs(baseline, candidate)
    baseline_row = rows[0]
    candidate_row = rows[1]
    return {
        "experiment_id": experiment_id,
        "baseline": baseline_row,
        "candidate": candidate_row,
        "derived": derived_metrics(baseline_row, candidate_row),
    }


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
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    if args.experiment:
        result = compare_runs_for_experiment(args.experiment, args.baseline, args.candidate)
        rows = [result["baseline"], result["candidate"]]
        json_payload: Any = result
    else:
        rows = compare_runs(args.baseline, args.candidate)
        json_payload = rows
    print_comparison(rows)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(json_payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
