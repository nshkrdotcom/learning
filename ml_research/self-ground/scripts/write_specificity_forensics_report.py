from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _group(rows: list[dict[str, Any]], key_fn, label: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(key_fn(row) or "unknown")].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        target = [_float(row.get("target_absolute_delta")) for row in group]
        control = [_float(row.get("control_absolute_delta")) for row in group]
        output.append(
            {
                label: key,
                "n_rows": len(group),
                "n_tasks": len({row.get("task_id") for row in group}),
                "target_delta_mean": _mean(target),
                "control_delta_mean": _mean(control),
                "specificity_gap_mean": _mean(target) - _mean(control),
                "control_dominant_rows": sum(
                    c > t for t, c in zip(target, control, strict=True)
                ),
            }
        )
    return output


def write_forensics(run_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    tasks_by_run: dict[str, dict[str, dict[str, Any]]] = {}
    for run_dir in run_dirs:
        run_name = run_dir.name
        tasks = {row.get("id"): row for row in _read_jsonl(run_dir / "behavioral_tasks.jsonl")}
        tasks_by_run[run_name] = tasks
        for row in _read_jsonl(run_dir / "behavioral_intervention_results.jsonl"):
            task = tasks.get(row.get("task_id"), {})
            metadata = task.get("metadata") or {}
            all_rows.append(
                {
                    **row,
                    "run_name": run_name,
                    "template_id": metadata.get("template_id")
                    or metadata.get("template_family")
                    or "unknown",
                    "token_pair": "/".join(row.get("target_tokens") or [])
                    + " vs "
                    + "/".join(row.get("foil_tokens") or []),
                }
            )

    family_rows = _group(
        all_rows,
        lambda row: f"{row.get('run_name')}::{row.get('family')}",
        "family",
    )
    token_rows = _group(
        all_rows,
        lambda row: f"{row.get('run_name')}::{row.get('token_pair')}",
        "token_pair",
    )
    template_rows = _group(
        all_rows,
        lambda row: f"{row.get('run_name')}::{row.get('template_id')}",
        "template",
    )
    control_rows = _group(
        all_rows,
        lambda row: (
            f"{row.get('run_name')}::"
            f"{row.get('control_suite', 'matched_non_negation_current')}"
        ),
        "control_suite",
    )
    feature_rows = _group(
        all_rows,
        lambda row: f"{row.get('run_name')}::{row.get('feature_set_label')}",
        "feature_set",
    )
    outlier_rows = sorted(
        [
            {
                "run_name": row.get("run_name"),
                "task_id": row.get("task_id"),
                "family": row.get("family"),
                "template_id": row.get("template_id"),
                "token_pair": row.get("token_pair"),
                "feature_set_label": row.get("feature_set_label"),
                "control_suite": row.get("control_suite"),
                "target_absolute_delta": row.get("target_absolute_delta"),
                "control_absolute_delta": row.get("control_absolute_delta"),
                "specificity_gap": _float(row.get("target_absolute_delta"))
                - _float(row.get("control_absolute_delta")),
            }
            for row in all_rows
        ],
        key=lambda row: float(row["specificity_gap"]),
    )

    _write_csv(
        out_dir / "task_outlier_table.csv",
        outlier_rows[:100],
        list(outlier_rows[0]) if outlier_rows else ["task_id"],
    )
    _write_csv(
        out_dir / "family_breakdown.csv",
        family_rows,
        list(family_rows[0]) if family_rows else ["family"],
    )
    _write_csv(
        out_dir / "token_pair_breakdown.csv",
        token_rows,
        list(token_rows[0]) if token_rows else ["token_pair"],
    )
    _write_csv(
        out_dir / "template_breakdown.csv",
        template_rows,
        list(template_rows[0]) if template_rows else ["template"],
    )
    _write_csv(
        out_dir / "control_suite_breakdown.csv",
        control_rows,
        list(control_rows[0]) if control_rows else ["control_suite"],
    )
    _write_csv(
        out_dir / "feature_breakdown.csv",
        feature_rows,
        list(feature_rows[0]) if feature_rows else ["feature_set"],
    )

    run_summaries = []
    for run_dir in run_dirs:
        report = _read_json(run_dir / "mechanism_report.json")
        summary_rows = [row for row in all_rows if row.get("run_name") == run_dir.name]
        target = [_float(row.get("target_absolute_delta")) for row in summary_rows]
        control = [_float(row.get("control_absolute_delta")) for row in summary_rows]
        run_summaries.append(
            {
                "run": str(run_dir),
                "claim_status": report.get("claim_status"),
                "target_delta_mean": _mean(target),
                "control_delta_mean": _mean(control),
                "specificity_gap_mean": _mean(target) - _mean(control),
                "n_rows": len(summary_rows),
            }
        )
    markdown = "# Specificity Forensics\n\n"
    for row in run_summaries:
        markdown += (
            f"- `{row['run']}`: status=`{row['claim_status']}`, "
            f"target=`{row['target_delta_mean']}`, control=`{row['control_delta_mean']}`, "
            f"gap=`{row['specificity_gap_mean']}`, rows=`{row['n_rows']}`\n"
        )
    markdown += """

## Interpretation

This report is artifact-only. It identifies where target/control specificity
fails or improves, but it does not rerun the model or prove feature
monosemanticity.
"""
    (out_dir / "forensics_summary.md").write_text(markdown, encoding="utf-8")
    return {"runs": run_summaries, "n_rows": len(all_rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Write artifact-only specificity forensics.")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summary = write_forensics([Path(item) for item in args.runs], Path(args.out))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
