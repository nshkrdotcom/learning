from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from scripts.inspect_claim_run import inspect_claim_run
except ModuleNotFoundError:
    try:
        from inspect_claim_run import inspect_claim_run
    except ModuleNotFoundError:
        import importlib.util

        inspect_path = Path(__file__).resolve().parent / "inspect_claim_run.py"
        spec = importlib.util.spec_from_file_location("inspect_claim_run", inspect_path)
        if spec is None or spec.loader is None:
            raise
        inspect_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inspect_module)
        inspect_claim_run = inspect_module.inspect_claim_run


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _baseline_pass_rate(run_dir: Path) -> float | None:
    rows = []
    path = run_dir / "baseline_task_scores.jsonl"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        return None
    return sum(1 for row in rows if bool(row.get("intended_direction_pass"))) / len(rows)


def _task_source(run_dir: Path) -> dict[str, Any]:
    return _read_json(run_dir / "task_source.json")


def _run_row(label: str, run_dir: Path) -> dict[str, Any]:
    summary = inspect_claim_run(run_dir, allow_missing=True)
    task_source = _task_source(run_dir)
    config = summary.get("config", {})
    metrics = summary.get("metrics", {})
    validation = summary.get("task_validation", {})
    return {
        "label": label,
        "run_dir": str(run_dir),
        "task_source": task_source.get("task_source") or config.get("task_source") or "generated",
        "task_source_id": task_source.get("task_source_id") or config.get("task_source_id"),
        "valid_tasks": validation.get("valid_tasks"),
        "valid_by_family": validation.get("valid_by_family"),
        "baseline_intended_direction_pass_rate": _baseline_pass_rate(run_dir),
        "claim_status": summary.get("claim", {}).get("claim_status"),
        "top_target_delta": metrics.get("top_target_delta"),
        "top_control_delta": metrics.get("top_control_delta"),
        "specificity_gap": metrics.get("specificity_gap"),
        "top_vs_control_ratio": metrics.get("top_vs_control_ratio"),
        "skipped_rows": summary.get("rows", {}).get("skipped_rows"),
        "limitations": summary.get("claim", {}).get("limitations", []),
        "blocker_reason": summary.get("claim", {}).get("blocker_reason"),
        "ok": summary.get("ok"),
    }


def _float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _interpretation(e002: dict[str, Any], e003: dict[str, Any]) -> str:
    e003_status = str(e003.get("claim_status") or "")
    e003_blocker = str(e003.get("blocker_reason") or "")
    if "task bank" in e003_blocker or "calibration" in e003_blocker:
        return "calibration_failed_to_build_task_suite"
    if e003_status == "blocked" or not e003.get("ok"):
        return "calibrated_run_blocked"
    if e003_status in {"candidate_evidence", "strong_candidate_evidence"}:
        return "candidate_evidence_under_calibrated_tasks"
    e002_spec = _float(e002.get("specificity_gap"))
    e003_spec = _float(e003.get("specificity_gap"))
    if e002_spec is not None and e003_spec is not None and e003_spec > e002_spec:
        return "calibration_fixed_task_suite_but_feature_specificity_still_failed"
    return "calibration_fixed_task_suite_but_feature_specificity_still_failed"


def compare_e002_e003(e002_dir: Path, e003_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    e002 = _run_row("E002_uncalibrated", e002_dir)
    e003 = _run_row("E003_calibrated", e003_dir)
    interpretation = _interpretation(e002, e003)

    rows = [e002, e003]
    _write_csv(
        out_dir / "comparison.csv",
        rows,
        [
            "label",
            "run_dir",
            "task_source",
            "task_source_id",
            "valid_tasks",
            "valid_by_family",
            "baseline_intended_direction_pass_rate",
            "claim_status",
            "top_target_delta",
            "top_control_delta",
            "specificity_gap",
            "top_vs_control_ratio",
            "skipped_rows",
            "blocker_reason",
        ],
    )

    family_rows: list[dict[str, Any]] = []
    for label, run_dir in [("E002_uncalibrated", e002_dir), ("E003_calibrated", e003_dir)]:
        validation = _read_json(run_dir / "behavioral_task_validation.json")
        summary = validation.get("summary", validation)
        valid_by_family = dict(summary.get("valid_by_family", {}))
        calibration = _read_json(run_dir / "source_calibration_summary.json")
        kept_by_family = dict(calibration.get("kept_by_family", {}))
        for family in sorted(set(valid_by_family) | set(kept_by_family)):
            family_rows.append(
                {
                    "run": label,
                    "family": family,
                    "valid_tasks": valid_by_family.get(family, ""),
                    "calibrated_kept_tasks": kept_by_family.get(family, ""),
                }
            )
    _write_csv(
        out_dir / "family_comparison.csv",
        family_rows,
        ["run", "family", "valid_tasks", "calibrated_kept_tasks"],
    )

    payload = {
        "e002": e002,
        "e003": e003,
        "interpretation_category": interpretation,
        "specificity_gap_delta": (
            (_float(e003.get("specificity_gap")) or 0.0)
            - (_float(e002.get("specificity_gap")) or 0.0)
        ),
        "top_vs_control_ratio_delta": (
            (_float(e003.get("top_vs_control_ratio")) or 0.0)
            - (_float(e002.get("top_vs_control_ratio")) or 0.0)
        ),
    }
    (out_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    claim_delta = f"""# E002 vs E003 Claim Delta

- interpretation category: `{interpretation}`
- E002 claim status: `{e002.get("claim_status")}`
- E003 claim status: `{e003.get("claim_status")}`
- E002 specificity gap: `{e002.get("specificity_gap")}`
- E003 specificity gap: `{e003.get("specificity_gap")}`
- specificity gap delta: `{payload["specificity_gap_delta"]}`

E003 uses a baseline-calibrated task source when its task_source artifact says
`task_source=file`. Any positive E003 claim is conditional on that calibrated
task bank and the current custom token-contrast evaluator.
"""
    (out_dir / "claim_delta.md").write_text(claim_delta, encoding="utf-8")
    (out_dir / "README.md").write_text(claim_delta, encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare E002 and calibrated E003 runs.")
    parser.add_argument("--e002", required=True)
    parser.add_argument("--e003", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = compare_e002_e003(Path(args.e002), Path(args.e003), Path(args.out))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
