from __future__ import annotations

from typing import Any

BLOCKER_PRIORITY = [
    "metadata_mismatch",
    "backend_untrusted",
    "artifact_incomplete",
    "preflight_failed",
    "off_manifold_intervention",
    "control_leaky",
    "density_matching_failed",
    "dictionary_interference",
    "neighbor_feature_interference",
    "self_repair_suspected",
    "insufficient_heldout_generalization",
    "insufficient_effect_size",
]


def diagnose_blockers(metrics: dict[str, Any], *, thresholds: dict[str, float]) -> dict[str, Any]:
    blockers: list[str] = []
    blocking_metrics: list[dict[str, Any]] = []
    target_delta = _float(metrics.get("target_delta"))
    matched_control_delta = _float(metrics.get("matched_control_delta"))
    family_min_gap = _float(metrics.get("family_min_gap"))
    specificity_gap = _float(metrics.get("specificity_gap"))
    control_leaky_ratio = thresholds.get("control_leaky_ratio", 0.8)

    if target_delta is None:
        blockers.append("artifact_incomplete")
        blocking_metrics.append({"name": "target_delta", "status": "missing"})
    if matched_control_delta is None:
        blockers.append("artifact_incomplete")
        blocking_metrics.append({"name": "matched_control_delta", "status": "missing"})

    if target_delta is not None and matched_control_delta is not None:
        if matched_control_delta >= target_delta * control_leaky_ratio:
            blockers.append("control_leaky")
            blocking_metrics.append(
                {
                    "name": "matched_control_delta",
                    "value": matched_control_delta,
                    "threshold": f"< {control_leaky_ratio} * target_delta",
                    "status": "fail",
                }
            )
    if family_min_gap is not None and family_min_gap <= 0:
        blockers.append("control_leaky")
        blocking_metrics.append(
            {
                "name": "family_min_gap",
                "value": family_min_gap,
                "threshold": "> 0",
                "status": "fail",
            }
        )
    if specificity_gap is not None and specificity_gap <= 0:
        blockers.append("specificity_gap_failed")

    deduped = []
    for blocker in blockers:
        if blocker not in deduped:
            deduped.append(blocker)
    primary = next((blocker for blocker in BLOCKER_PRIORITY if blocker in deduped), None)
    return {
        "blockers": deduped,
        "primary_blocker": primary,
        "blocking_metrics": blocking_metrics,
    }


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
