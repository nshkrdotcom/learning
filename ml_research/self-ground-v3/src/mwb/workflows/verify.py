from __future__ import annotations

from typing import Any

from mwb.domain.objects import PredictionLock, VerificationRun
from mwb.refs import stable_ref


def run_verify(
    hypothesis_payload: dict[str, Any],
    *,
    prediction_lock: PredictionLock | dict[str, Any] | None,
    diagnostic_only: bool,
    dry_run: bool,
) -> VerificationRun:
    hypothesis_ref = str(hypothesis_payload["wb_ref"])
    if prediction_lock is None and not diagnostic_only:
        return VerificationRun(
            wb_ref=stable_ref("ver", hypothesis_ref, "prediction_lock_missing"),
            hypothesis_ref=hypothesis_ref,
            prediction_lock_ref=None,
            status="blocked",
            evidence_posture="blocked",
            metrics={},
            metadata={"blockers": ["prediction_lock_missing"], "dry_run": dry_run},
            parents=[hypothesis_ref],
        )

    lock_ref = None
    if isinstance(prediction_lock, PredictionLock):
        lock_ref = prediction_lock.wb_ref
    elif isinstance(prediction_lock, dict):
        lock_ref = prediction_lock.get("wb_ref")

    evidence_posture = "diagnostic_only" if diagnostic_only else "claim_bearing"
    status = "dry_run" if dry_run else "planned"
    parents = [hypothesis_ref]
    if lock_ref:
        parents.append(str(lock_ref))
    return VerificationRun(
        wb_ref=stable_ref("ver", hypothesis_ref, lock_ref or "diagnostic", status),
        hypothesis_ref=hypothesis_ref,
        prediction_lock_ref=lock_ref,
        status=status,
        evidence_posture=evidence_posture,
        metrics={},
        metadata={"dry_run": dry_run},
        parents=parents,
    )

