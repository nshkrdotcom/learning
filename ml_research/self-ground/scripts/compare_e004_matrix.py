from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_eval_dir(path: Path) -> dict[str, str]:
    config = _read_json(path / "config.json")
    return {
        "layer": str(config.get("hook_point") or ""),
        "sae_release": str(config.get("sae_release") or ""),
        "sae_id": str(config.get("sae_id") or ""),
        "feature_selection_mode": str(config.get("feature_selection_mode") or ""),
        "operation": ",".join(config.get("operations") or []),
        "control_suite": str(config.get("control_suite") or ""),
    }


def _completed_row(path: Path) -> dict[str, Any]:
    report = _read_json(path / "mechanism_report.json")
    config_bits = _parse_eval_dir(path)
    summary_rows = _read_csv(path / "behavioral_summary.csv")
    top_all = [
        row
        for row in summary_rows
        if row.get("feature_set_label") == "top" and row.get("family") == "__all__"
    ]
    family_top = [
        row
        for row in summary_rows
        if row.get("feature_set_label") == "top" and row.get("family") != "__all__"
    ]
    top = top_all[0] if top_all else {}
    target = _float(top.get("target_absolute_delta_mean"))
    control = _float(top.get("control_absolute_delta_mean"))
    gap = _float(top.get("specificity_gap_mean"))
    control_suite_gaps = [
        value for row in top_all if (value := _float(row.get("specificity_gap_mean"))) is not None
    ]
    family_gaps = [
        value
        for row in family_top
        if (value := _float(row.get("specificity_gap_mean"))) is not None
    ]
    validation = _read_json(path / "behavioral_task_validation.json").get("summary", {})
    skipped = _read_json(path / "skipped_behavioral_rows.json")
    density_gaps = [
        _float(row.get("specificity_gap_mean"))
        for row in summary_rows
        if str(row.get("feature_set_label", "")).startswith("density_matched_seed_")
        and row.get("family") == "__all__"
    ]
    return {
        **config_bits,
        "run_status": "completed",
        "claim_status": report.get("claim_status"),
        "valid_tasks": validation.get("valid_tasks"),
        "behavioral_rows": len(
            (path / "behavioral_intervention_results.jsonl").read_text().splitlines()
        )
        if (path / "behavioral_intervention_results.jsonl").exists()
        else 0,
        "skipped_rows": skipped.get("n_skipped_rows"),
        "baseline_pass_rate": (
            report.get("feature_sets", [{}])[0].get("intended_direction_pass_rate")
            if report.get("feature_sets")
            else None
        ),
        "top_target_delta": target,
        "top_control_delta": control,
        "specificity_gap": gap,
        "top_vs_control_ratio": (
            target / control if target is not None and control and control > 0 else None
        ),
        "density_control_gap": min([gap for gap in density_gaps if gap is not None], default=None),
        "multi_control_min_gap": min(control_suite_gaps, default=None),
        "family_min_specificity_gap": min(family_gaps, default=None),
        "passes_all_controls": bool(control_suite_gaps) and min(control_suite_gaps) > 0,
        "limitations": " | ".join(report.get("limitations", [])),
        "artifact_paths": str(path),
    }


def _blocked_row(path: Path) -> dict[str, Any]:
    blocker = _read_json(path / "CELL_BLOCKED.json") or _read_json(path / "BLOCKED.json")
    return {
        **_parse_eval_dir(path),
        "run_status": "blocked",
        "claim_status": "blocked",
        "valid_tasks": None,
        "behavioral_rows": 0,
        "skipped_rows": None,
        "baseline_pass_rate": None,
        "top_target_delta": None,
        "top_control_delta": None,
        "specificity_gap": None,
        "top_vs_control_ratio": None,
        "density_control_gap": None,
        "multi_control_min_gap": None,
        "family_min_specificity_gap": None,
        "passes_all_controls": False,
        "limitations": blocker.get("reason"),
        "artifact_paths": str(path),
    }


def _candidate_allowed(row: dict[str, Any]) -> bool:
    return (
        row.get("claim_status") in {"candidate_evidence", "strong_candidate_evidence"}
        and bool(row.get("passes_all_controls"))
        and (row.get("family_min_specificity_gap") is not None)
        and float(row["family_min_specificity_gap"]) > 0
    )


def compare_matrix(matrix_root: Path, e003: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    eval_root = matrix_root / "eval"
    for path in sorted(eval_root.iterdir() if eval_root.exists() else []):
        if not path.is_dir():
            continue
        if (path / "mechanism_report.json").exists():
            rows.append(_completed_row(path))
        elif (path / "CELL_BLOCKED.json").exists() or (path / "BLOCKED.json").exists():
            rows.append(_blocked_row(path))

    e003_row = (
        _completed_row(e003)
        if (e003 / "mechanism_report.json").exists()
        else _blocked_row(e003)
    )
    fieldnames = [
        "layer",
        "sae_release",
        "sae_id",
        "feature_selection_mode",
        "operation",
        "control_suite",
        "run_status",
        "claim_status",
        "valid_tasks",
        "behavioral_rows",
        "skipped_rows",
        "baseline_pass_rate",
        "top_target_delta",
        "top_control_delta",
        "specificity_gap",
        "top_vs_control_ratio",
        "density_control_gap",
        "multi_control_min_gap",
        "family_min_specificity_gap",
        "passes_all_controls",
        "limitations",
        "artifact_paths",
    ]
    _write_csv(out_dir / "matrix_summary.csv", rows, fieldnames)
    _write_json(out_dir / "matrix_summary.json", rows)
    completed = [row for row in rows if row["run_status"] == "completed"]
    blocked = [row for row in rows if row["run_status"] == "blocked"]
    best = sorted(
        completed,
        key=lambda row: (
            -float(row["specificity_gap"] or -999.0),
            str(row["layer"]),
            str(row["feature_selection_mode"]),
        ),
    )
    _write_csv(out_dir / "best_runs_by_specificity.csv", best[:10], fieldnames)
    _write_csv(out_dir / "blocked_runs.csv", blocked, fieldnames)
    best_by_family = sorted(
        completed,
        key=lambda row: (
            -float(row["family_min_specificity_gap"] or -999.0),
            str(row["layer"]),
        ),
    )
    _write_csv(out_dir / "best_runs_by_family.csv", best_by_family[:10], fieldnames)
    candidates = [row for row in completed if _candidate_allowed(row)]
    if candidates:
        interpretation = "candidate_evidence_under_qualified_e004_cell"
    elif completed:
        interpretation = "current_sae_model_layer_search_insufficient"
    else:
        interpretation = "all_e004_cells_blocked"
    payload = {
        "matrix_root": str(matrix_root),
        "e003": str(e003),
        "attempted_cells": len(rows),
        "completed_cells": len(completed),
        "blocked_cells": len(blocked),
        "candidate_cells": len(candidates),
        "interpretation": interpretation,
        "best_run": best[0] if best else None,
        "e003_specificity_gap": e003_row.get("specificity_gap"),
    }
    _write_json(out_dir / "comparison.json", payload)
    adjudication = f"""# E004 Claim Adjudication

- attempted cells: `{len(rows)}`
- completed cells: `{len(completed)}`
- blocked cells: `{len(blocked)}`
- candidate cells: `{len(candidates)}`
- interpretation: `{interpretation}`

## Best Run

```json
{json.dumps(best[0] if best else None, indent=2, sort_keys=True)}
```

## Unsupported Claims

- Broad negation mechanism discovery is not supported.
- Upstream SAEBench/RAVEL benchmark evidence is not established.
- Layer-general or model-general conclusions are not supported by this matrix.
- Monosemantic feature claims are not supported.
"""
    (out_dir / "claim_adjudication.md").write_text(adjudication, encoding="utf-8")
    (out_dir / "README.md").write_text(adjudication, encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare E004 specificity rescue matrix runs.")
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--e003", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = compare_matrix(Path(args.matrix_root), Path(args.e003), Path(args.out))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
