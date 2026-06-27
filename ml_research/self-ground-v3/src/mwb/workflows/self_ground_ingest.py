from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from mwb.domain.objects import BlockerReport
from mwb.project import Project, ProjectManager
from mwb.refs import slugify, stable_ref
from mwb.time import utc_now
from mwb.workflows.blockers import diagnose_blockers
from mwb.workflows.cards import card_from_run, write_card
from mwb.workflows.next_probe import build_next_probe, write_next_probe

SUMMARY_COLUMNS = [
    "layer",
    "feature_selection_mode",
    "claim_status",
    "top_target_delta",
    "top_control_delta",
    "specificity_gap",
    "family_min_specificity_gap",
    "multi_control_min_gap",
]
FORENSICS_BREAKDOWN_COLUMNS = [
    "n_rows",
    "n_tasks",
    "target_delta_mean",
    "control_delta_mean",
    "specificity_gap_mean",
    "control_dominant_rows",
]

REQUIRED_ARTIFACTS = {
    "capability.json": [],
    "matrix_run_summary.json": [
        "status",
        "attempted_cells",
        "completed_cells",
        "layers",
        "feature_selection_modes",
        "operations",
        "control_suite",
    ],
    "comparison/comparison.json": ["best_run", "interpretation"],
    "comparison/matrix_summary.json": SUMMARY_COLUMNS,
    "comparison/matrix_summary.csv": SUMMARY_COLUMNS,
    "comparison/best_runs_by_family.csv": SUMMARY_COLUMNS,
    "comparison/best_runs_by_specificity.csv": SUMMARY_COLUMNS,
    "comparison/blocked_runs.csv": SUMMARY_COLUMNS,
    "comparison/claim_adjudication.md": [],
    "forensics/control_suite_breakdown.csv": ["control_suite", *FORENSICS_BREAKDOWN_COLUMNS],
    "forensics/family_breakdown.csv": ["family", *FORENSICS_BREAKDOWN_COLUMNS],
    "forensics/feature_breakdown.csv": ["feature_set", *FORENSICS_BREAKDOWN_COLUMNS],
    "forensics/task_outlier_table.csv": [
        "run_name",
        "task_id",
        "family",
        "template_id",
        "token_pair",
        "feature_set_label",
        "control_suite",
        "target_absolute_delta",
        "control_absolute_delta",
        "specificity_gap",
    ],
    "forensics/template_breakdown.csv": ["template", *FORENSICS_BREAKDOWN_COLUMNS],
    "forensics/token_pair_breakdown.csv": ["token_pair", *FORENSICS_BREAKDOWN_COLUMNS],
    "forensics/forensics_summary.md": [],
}


def ingest_self_ground_run(source: Path, *, project: Project | None = None) -> Path:
    project = project or ProjectManager.discover_or_create()
    source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"SELF-GROUND run directory not found: {source}")

    artifact_status = validate_self_ground_artifacts(source)
    missing = [name for name, status in artifact_status.items() if status["status"] != "present"]
    if missing:
        raise ValueError(f"SELF-GROUND artifact set is incomplete: {', '.join(missing)}")

    capability = _read_json(source / "capability.json")
    matrix_summary = _read_json(source / "matrix_run_summary.json")
    comparison = _read_json(source / "comparison" / "comparison.json")
    matrix_rows = _read_json(source / "comparison" / "matrix_summary.json")
    if not isinstance(matrix_rows, list):
        raise ValueError("comparison/matrix_summary.json must contain a list of row objects")

    best_run = comparison.get("best_run")
    if not isinstance(best_run, dict):
        best_run = _select_best_row(matrix_rows)
    if not best_run:
        raise ValueError("SELF-GROUND comparison has no best_run or summary rows")

    run_ref = f"run_self_ground_{slugify(source.name)}"
    run_dir = project.mechanism_dir / "runs" / run_ref
    run_dir.mkdir(parents=True, exist_ok=True)

    status = _claim_status(best_run, comparison)
    metrics = _control_metrics(best_run)
    blockers = diagnose_blockers(metrics, thresholds={"control_leaky_ratio": 0.8})
    manifest = _run_manifest(
        run_ref=run_ref,
        source=source,
        status=status,
        artifact_status=artifact_status,
        matrix_summary=matrix_summary,
        comparison=comparison,
        best_run=best_run,
        capability=capability,
    )

    _write_json(run_dir / "run_manifest.json", manifest)
    _write_json(run_dir / "control_metrics.json", metrics)
    _write_blocker_report(run_dir, run_ref, blockers)
    plan = build_next_probe(
        {
            "run_ref": run_ref,
            "status": status,
            "metrics": metrics,
            "blockers": blockers["blockers"],
            "tried_axes": manifest["tried_axes"],
            "available_axes": manifest["available_axes"],
            "backend_capabilities": manifest["backend_capabilities"],
        }
    )
    write_next_probe(run_dir, plan)
    card = card_from_run(run_dir)
    write_card(run_dir, card, mechanism_dir=project.mechanism_dir)
    return run_dir


def validate_self_ground_artifacts(source: Path) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for relative, required_columns in REQUIRED_ARTIFACTS.items():
        path = source / relative
        if not path.exists():
            statuses[relative] = {"status": "missing", "required_columns": required_columns}
            continue
        status: dict[str, Any] = {
            "status": "present",
            "path": str(path),
            "required_columns": required_columns,
        }
        if required_columns:
            observed, row_count = _observed_fields(path)
            missing = [column for column in required_columns if column not in observed]
            status["observed_columns"] = sorted(observed)
            status["missing_columns"] = missing
            status["row_count"] = row_count
            if missing:
                status["status"] = "missing_columns"
        elif path.suffix in {".csv", ".json"}:
            _observed, row_count = _observed_fields(path)
            status["row_count"] = row_count
        statuses[relative] = status
    return statuses


def _run_manifest(
    *,
    run_ref: str,
    source: Path,
    status: str,
    artifact_status: dict[str, dict[str, Any]],
    matrix_summary: dict[str, Any],
    comparison: dict[str, Any],
    best_run: dict[str, Any],
    capability: dict[str, Any],
) -> dict[str, Any]:
    tried_layers = _string_list(matrix_summary.get("layers"))
    tried_modes = _string_list(matrix_summary.get("feature_selection_modes"))
    tried_operations = _string_list(matrix_summary.get("operations"))
    return {
        "run_ref": run_ref,
        "source_kind": "self_ground_e004",
        "source_path": str(source),
        "ingested_at": utc_now(),
        "status": status,
        "evidence_posture": "diagnostic_insufficient",
        "claim_bearing": False,
        "source_artifacts": artifact_status,
        "summary": {
            "matrix_status": matrix_summary.get("status"),
            "attempted_cells": matrix_summary.get("attempted_cells"),
            "completed_cells": matrix_summary.get("completed_cells"),
            "blocked_cells": matrix_summary.get("blocked_cells"),
            "candidate_cells": comparison.get("candidate_cells"),
            "interpretation": comparison.get("interpretation"),
            "best_run": _public_best_run(best_run),
        },
        "tried_axes": {
            "layers": tried_layers,
            "feature_selection_modes": tried_modes,
            "operations": tried_operations,
            "control_suites": _string_list(matrix_summary.get("control_suite")),
        },
        "available_axes": {
            "layers": _neighbor_layers(tried_layers),
            "feature_selection_modes": tried_modes,
            "operations": tried_operations,
            "patch_modes": ["delta", "direct"],
            "control_suites": ["multi_control", "hard_negative_control", "matched_control"],
        },
        "backend_capabilities": {
            "cuda_available": capability.get("cuda_available"),
            "torch_version": capability.get("torch_version"),
            "direct_patch": True,
            "source_backend": "self-ground",
        },
        "metadata": {
            "matrix_root": comparison.get("matrix_root"),
            "adjudication": "comparison/claim_adjudication.md",
        },
    }


def _control_metrics(best_run: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "target_delta": _float(best_run.get("top_target_delta")),
        "matched_control_delta": _float(best_run.get("top_control_delta")),
        "specificity_gap": _float(best_run.get("specificity_gap")),
        "family_min_gap": _float(best_run.get("family_min_specificity_gap")),
        "multi_control_min_gap": _float(best_run.get("multi_control_min_gap")),
        "density_control_gap": _float(best_run.get("density_control_gap")),
        "top_vs_control_ratio": _float(best_run.get("top_vs_control_ratio")),
        "baseline_pass_rate": _float(best_run.get("baseline_pass_rate")),
        "passes_all_controls": _bool(best_run.get("passes_all_controls")),
        "valid_tasks": _int(best_run.get("valid_tasks")),
        "behavioral_rows": _int(best_run.get("behavioral_rows")),
        "source_columns": {
            "target_delta": "top_target_delta",
            "matched_control_delta": "top_control_delta",
            "family_min_gap": "family_min_specificity_gap",
            "multi_control_min_gap": "multi_control_min_gap",
        },
        "source_run": _public_best_run(best_run),
    }
    return {key: value for key, value in metrics.items() if value is not None}


def _write_blocker_report(
    run_dir: Path, run_ref: str, blockers: dict[str, Any]
) -> BlockerReport:
    report = BlockerReport(
        wb_ref=stable_ref("blocker", run_ref, blockers),
        run_ref=run_ref,
        blockers=list(blockers["blockers"]),
        primary_blocker=blockers["primary_blocker"],
        blocking_metrics=list(blockers["blocking_metrics"]),
        parents=[run_ref],
    )
    _write_json(run_dir / "blocker_report.json", report.model_dump(mode="json"))
    return report


def _observed_fields(path: Path) -> tuple[set[str], int | None]:
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            row_count = sum(1 for _row in reader)
            return {str(name) for name in reader.fieldnames or []}, row_count
    payload = _read_json(path)
    if isinstance(payload, list):
        fields: set[str] = set()
        for row in payload:
            if isinstance(row, dict):
                fields.update(str(key) for key in row)
        return fields, len(payload)
    if isinstance(payload, dict):
        fields = {str(key) for key in payload}
        best_run = payload.get("best_run")
        if isinstance(best_run, dict):
            fields.update(str(key) for key in best_run)
        return fields, 1
    return set(), None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _select_best_row(rows: list[Any]) -> dict[str, Any]:
    row_dicts = [row for row in rows if isinstance(row, dict)]
    if not row_dicts:
        return {}
    return max(row_dicts, key=lambda row: (_float(row.get("specificity_gap")) or float("-inf")))


def _claim_status(best_run: dict[str, Any], comparison: dict[str, Any]) -> str:
    claim_status = str(best_run.get("claim_status") or "").strip()
    if claim_status:
        return claim_status
    interpretation = str(comparison.get("interpretation") or "").lower()
    if "insufficient" in interpretation:
        return "insufficient_evidence"
    return "insufficient_evidence"


def _public_best_run(best_run: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "layer",
        "sae_release",
        "sae_id",
        "feature_selection_mode",
        "operation",
        "control_suite",
        "run_status",
        "claim_status",
        "passes_all_controls",
        "limitations",
        "artifact_paths",
    ]
    return {key: best_run[key] for key in keys if key in best_run}


def _neighbor_layers(layers: list[str]) -> list[str]:
    observed = list(dict.fromkeys(layers))
    numbers = [_layer_number(layer) for layer in observed]
    if not numbers:
        return observed
    available = {layer for layer in observed}
    for number in numbers:
        if number > 0:
            available.add(f"blocks.{number - 1}.hook_resid_post")
        available.add(f"blocks.{number + 1}.hook_resid_post")
    return sorted(available, key=lambda layer: (_layer_number(layer), layer))


def _layer_number(layer: str) -> int:
    parts = layer.split(".")
    try:
        block_index = parts.index("blocks")
    except ValueError:
        return 10_000
    try:
        return int(parts[block_index + 1])
    except (IndexError, ValueError):
        return 10_000


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)
