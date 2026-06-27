from __future__ import annotations

import json
from functools import reduce
from itertools import product
from operator import mul
from pathlib import Path
from typing import Any

from mwb.project import Project
from mwb.refs import stable_ref
from mwb.time import utc_now


def parse_axes(axis: list[str]) -> dict:
    axes: dict[str, list[str]] = {}
    for raw in axis:
        if "=" not in raw:
            raise ValueError(f"axis must be name=value[,value...]: {raw}")
        name, values = raw.split("=", 1)
        parsed_values = [value for value in values.split(",") if value]
        if not name or not parsed_values:
            raise ValueError(f"axis must be name=value[,value...]: {raw}")
        axes[name] = parsed_values
    matrix_size = reduce(mul, (len(values) for values in axes.values()), 1)
    return {
        "axis_source": "cli",
        "axes": axes,
        "inherited_axes": {},
        "matrix_semantics": "cross_product",
        "matrix_size": matrix_size,
    }


def write_sweep_run(
    *,
    project: Project,
    hypothesis_payload: dict[str, Any],
    config: dict[str, Any],
    dry_run: bool,
) -> tuple[Path, dict[str, Any]]:
    hypothesis_ref = str(hypothesis_payload["wb_ref"])
    run_ref = stable_ref(
        "run",
        "sweep",
        hypothesis_ref,
        config["axes"],
        "dry_run" if dry_run else "planned",
    )
    run_dir = project.mechanism_dir / "runs" / run_ref
    run_dir.mkdir(parents=True, exist_ok=True)

    sweep_config = {
        **config,
        "source_hypothesis_ref": hypothesis_ref,
        "dry_run": dry_run,
        "run_ref": run_ref,
        "run_dir": str(run_dir),
        "created_at": utc_now(),
    }
    manifest = {
        "run_ref": run_ref,
        "source_kind": "mwb_sweep",
        "source_hypothesis_ref": hypothesis_ref,
        "status": "dry_run" if dry_run else "planned",
        "claim_bearing": False,
        "evidence_posture": "diagnostic_only" if dry_run else "planned_not_executed",
        "tried_axes": _plural_axes(config["axes"]),
        "available_axes": _plural_axes(config["axes"]),
        "backend_capabilities": {"direct_patch": "direct" in config["axes"].get("patch_mode", [])},
        "created_at": utc_now(),
    }
    combinations = _axis_combinations(config["axes"])
    receipts = [_receipt(run_ref, hypothesis_ref, combo, dry_run=dry_run) for combo in combinations]
    results = [
        _verification_result(run_ref, hypothesis_ref, combo, dry_run=dry_run)
        for combo in combinations
    ]
    blocker_report = {
        "wb_ref": stable_ref("blocker", run_ref, "dry_run_no_claim_evidence"),
        "wb_type": "BlockerReport",
        "run_ref": run_ref,
        "blockers": ["artifact_incomplete"],
        "primary_blocker": "artifact_incomplete",
        "blocking_metrics": [
            {
                "name": "causal_execution",
                "status": "not_run",
                "reason": "Sweep was planned in dry-run mode.",
            }
        ],
        "parents": [run_ref],
    }

    _write_json(run_dir / "sweep_config.json", sweep_config)
    _write_json(run_dir / "run_manifest.json", manifest)
    _write_json(run_dir / "control_metrics.json", {})
    _write_json(run_dir / "blocker_report.json", blocker_report)
    _write_jsonl(run_dir / "intervention_receipts.jsonl", receipts)
    _write_jsonl(run_dir / "verification_results.jsonl", results)
    return run_dir, {
        **sweep_config,
        "status": manifest["status"],
        "claim_bearing": False,
    }


def _axis_combinations(axes: dict[str, list[str]]) -> list[dict[str, str]]:
    if not axes:
        return [{}]
    names = list(axes)
    return [
        dict(zip(names, values, strict=True))
        for values in product(*(axes[name] for name in names))
    ]


def _receipt(
    run_ref: str,
    hypothesis_ref: str,
    combo: dict[str, str],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "receipt_ref": stable_ref("receipt", run_ref, combo),
        "run_ref": run_ref,
        "hypothesis_ref": hypothesis_ref,
        "axis_values": combo,
        "status": "dry_run" if dry_run else "planned",
        "backend_executed": False,
        "claim_bearing": False,
        "created_at": utc_now(),
    }


def _verification_result(
    run_ref: str,
    hypothesis_ref: str,
    combo: dict[str, str],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "result_ref": stable_ref("ver", run_ref, combo),
        "run_ref": run_ref,
        "hypothesis_ref": hypothesis_ref,
        "axis_values": combo,
        "status": "dry_run" if dry_run else "planned",
        "evidence_posture": "diagnostic_only" if dry_run else "planned_not_executed",
        "claim_bearing": False,
        "metrics": {},
        "blockers": ["artifact_incomplete"],
        "created_at": utc_now(),
    }


def _plural_axes(axes: dict[str, list[str]]) -> dict[str, list[str]]:
    pluralized = dict(axes)
    aliases = {
        "layer": "layers",
        "patch_mode": "patch_modes",
        "control_family": "control_families",
        "operation": "operations",
    }
    for singular, plural in aliases.items():
        if singular in axes:
            pluralized[plural] = list(axes[singular])
    return pluralized


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
