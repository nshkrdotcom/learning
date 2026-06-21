from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from scripts.inspect_claim_run import inspect_claim_run
except ModuleNotFoundError:  # Direct script execution from the scripts directory path.
    from inspect_claim_run import inspect_claim_run


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def compare_variants(run_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        inspection = inspect_claim_run(run_dir, allow_missing=True)
        config = _read_json(run_dir / "config.json")
        calibration = _read_json(run_dir / "task_calibration_result.json")
        row = {
            "run_dir": str(run_dir),
            "task_calibration_mode": config.get("task_calibration_mode", "none"),
            "feature_selection_mode": config.get("feature_selection_mode", "top"),
            "tasks_before": calibration.get(
                "total_tasks_before",
                inspection.get("task_validation", {}).get("valid_tasks"),
            ),
            "tasks_after": calibration.get(
                "total_tasks_after",
                inspection.get("task_validation", {}).get("valid_tasks"),
            ),
            "valid_by_family_after": json.dumps(
                calibration.get(
                    "valid_by_family_after",
                    inspection.get("task_validation", {}).get("valid_by_family", {}),
                ),
                sort_keys=True,
            ),
            "claim_status": inspection.get("claim", {}).get("claim_status"),
            "top_target_delta": inspection.get("metrics", {}).get("top_target_delta"),
            "top_control_delta": inspection.get("metrics", {}).get("top_control_delta"),
            "specificity_gap": inspection.get("metrics", {}).get("specificity_gap"),
            "top_vs_control_ratio": inspection.get("metrics", {}).get("top_vs_control_ratio"),
            "skipped_rows": inspection.get("rows", {}).get("skipped_rows"),
            "limitations": json.dumps(inspection.get("claim", {}).get("limitations", [])),
        }
        rows.append(row)
    fieldnames = [
        "run_dir",
        "task_calibration_mode",
        "feature_selection_mode",
        "tasks_before",
        "tasks_after",
        "valid_by_family_after",
        "claim_status",
        "top_target_delta",
        "top_control_delta",
        "specificity_gap",
        "top_vs_control_ratio",
        "skipped_rows",
        "limitations",
    ]
    with (out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    payload = {
        "runs": rows,
        "interpretation": {
            "candidate_runs": [
                row["run_dir"]
                for row in rows
                if row["claim_status"] in {"candidate_evidence", "strong_candidate_evidence"}
            ],
            "blocked_runs": [row["run_dir"] for row in rows if row["claim_status"] == "blocked"],
            "insufficient_runs": [
                row["run_dir"] for row in rows if row["claim_status"] == "insufficient_evidence"
            ],
        },
    }
    (out_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    readme = "# E002 Variant Comparison\n\n"
    readme += (
        "This artifact compares bounded calibrated E002 variants from existing "
        "run artifacts.\n\n"
    )
    for row in rows:
        readme += (
            f"- `{row['run_dir']}`: status `{row['claim_status']}`, "
            f"specificity `{row['specificity_gap']}`\n"
        )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare bounded calibrated E002 variants.")
    parser.add_argument("--out", required=True)
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()
    payload = compare_variants([Path(item) for item in args.run_dirs], Path(args.out))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
