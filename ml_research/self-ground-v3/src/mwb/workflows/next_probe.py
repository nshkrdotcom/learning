from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from mwb.domain.objects import NextProbePlan
from mwb.refs import stable_ref
from mwb.workflows.blockers import diagnose_blockers

REQUIRED_FIELDS = ["run_ref", "status", "metrics"]


def build_next_probe(payload: dict[str, Any]) -> NextProbePlan:
    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        return NextProbePlan(
            wb_ref=stable_ref("next", payload.get("run_ref", "unknown"), "artifact_incomplete"),
            source_run_ref=str(payload.get("run_ref", "unknown")),
            status=str(payload.get("status", "blocked")),
            diagnosis={"primary": "artifact_incomplete", "secondary": []},
            recommendation={
                "kind": "recommendation_without_command",
                "rationale": "Required next-probe input fields are missing.",
            },
            missing_fields=missing,
        )

    metrics = payload.get("metrics", {})
    blockers = list(payload.get("blockers") or [])
    if not blockers:
        blockers = diagnose_blockers(metrics, thresholds={"control_leaky_ratio": 0.8})["blockers"]
    primary = _primary(blockers)
    tried_axes = payload.get("tried_axes", {})
    available_axes = payload.get("available_axes", {})
    recommendation = _recommend(primary, payload["run_ref"], tried_axes, available_axes)

    return NextProbePlan(
        wb_ref=stable_ref("next", payload["run_ref"], primary, recommendation),
        source_run_ref=str(payload["run_ref"]),
        status=str(payload["status"]),
        diagnosis={"primary": primary, "secondary": [b for b in blockers if b != primary]},
        recommendation=recommendation,
        claim_implication={
            "specificity": "blocked" if primary == "control_leaky" else "unknown",
            "mechanism": "blocked",
            "allowed_language": ["candidate associated with"],
            "blocked_language": ["implements"],
        },
        parents=[str(payload["run_ref"])],
    )


def load_next_probe_payload(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    run_manifest = _read_json(path / "run_manifest.json")
    control_metrics = _read_json(path / "control_metrics.json")
    blocker_report = _read_json(path / "blocker_report.json")
    payload = {
        "run_ref": run_manifest.get("run_ref", path.name),
        "status": run_manifest.get("status", "insufficient_evidence"),
        "metrics": control_metrics,
        "blockers": blocker_report.get("blockers", []),
        "tried_axes": run_manifest.get("tried_axes", {}),
        "available_axes": run_manifest.get("available_axes", {}),
        "backend_capabilities": run_manifest.get("backend_capabilities", {}),
    }
    if not payload["blockers"] and control_metrics:
        payload["blockers"] = diagnose_blockers(
            control_metrics,
            thresholds={"control_leaky_ratio": 0.8},
        )["blockers"]
    return payload


def write_next_probe(run_dir: Path, plan: NextProbePlan) -> None:
    payload = plan.model_dump(mode="json")
    (run_dir / "next_probe.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    yaml = YAML()
    with (run_dir / "next_probe.yaml").open("w", encoding="utf-8") as handle:
        yaml.dump(payload, handle)
    (run_dir / "next_probe.md").write_text(
        "\n".join(
            [
                f"# Next Probe: {plan.source_run_ref}",
                "",
                f"Primary diagnosis: `{plan.diagnosis['primary']}`",
                f"Recommendation: `{plan.recommendation['kind']}`",
                "",
                plan.recommendation.get("rationale", ""),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _primary(blockers: list[str]) -> str:
    if "metadata_mismatch" in blockers:
        return "metadata_mismatch"
    if "backend_untrusted" in blockers:
        return "backend_untrusted"
    if "artifact_incomplete" in blockers:
        return "artifact_incomplete"
    if "control_leaky" in blockers:
        return "control_leaky"
    return blockers[0] if blockers else "insufficient_effect_size"


def _recommend(
    primary: str,
    run_ref: str,
    tried_axes: dict[str, list[str]],
    available_axes: dict[str, list[str]],
) -> dict[str, Any]:
    if primary == "artifact_incomplete":
        return {
            "kind": "recommendation_without_command",
            "rationale": "Complete missing artifacts before choosing a scientific next probe.",
        }
    if primary == "control_leaky":
        untried_layers = _untried("layers", tried_axes, available_axes)
        untried_patch_modes = _untried("patch_modes", tried_axes, available_axes)
        if untried_layers:
            layer = untried_layers[0]
            return {
                "kind": "smallest_axis_extension",
                "rationale": (
                    "Controls moved too much; try the smallest untried adjacent layer axis."
                ),
                "command": f"uv run mwb sweep {run_ref} --axis layer={layer} --dry-run",
            }
        if untried_patch_modes and "direct" in untried_patch_modes:
            return {
                "kind": "switch_patch_mode",
                "rationale": (
                    "Controls moved too much; try direct patch mode if backend supports it."
                ),
                "command": f"uv run mwb sweep {run_ref} --axis patch_mode=direct --dry-run",
            }
        return {
            "kind": "refresh_controls",
            "rationale": "Controls moved too much and no smaller untried axis is available.",
        }
    return {
        "kind": "heldout_generalization",
        "rationale": "Run a scoped follow-up before strengthening the claim.",
    }


def _untried(
    name: str,
    tried_axes: dict[str, list[str]],
    available_axes: dict[str, list[str]],
) -> list[str]:
    tried = {str(value) for value in tried_axes.get(name, [])}
    return [str(value) for value in available_axes.get(name, []) if str(value) not in tried]
