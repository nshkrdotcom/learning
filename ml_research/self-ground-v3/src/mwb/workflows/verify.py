from __future__ import annotations

from typing import Any

from mwb.domain.objects import PredictionLock, VerificationRun
from mwb.refs import stable_ref
from mwb.static_compiler import StaticCompiler, has_static_compiler_payload


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

    if not diagnostic_only:
        if not has_static_compiler_payload(hypothesis_payload):
            return _blocked_verification(
                hypothesis_ref,
                ["static_compiler_missing"],
                dry_run=dry_run,
                prediction_lock_ref=lock_ref,
            )
        static_report = StaticCompiler().compile_payload(hypothesis_payload)
        if static_report.status == "fail":
            return _blocked_verification(
                hypothesis_ref,
                ["static_compiler_failed", *static_report.blockers],
                dry_run=dry_run,
                static_report_ref=static_report.wb_ref,
                plausibility_gate=static_report.plausibility_gate,
                prediction_lock_ref=lock_ref,
            )

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


def _blocked_verification(
    hypothesis_ref: str,
    blockers: list[str],
    *,
    dry_run: bool,
    static_report_ref: str | None = None,
    plausibility_gate: str | None = None,
    prediction_lock_ref: str | None = None,
) -> VerificationRun:
    metadata: dict[str, Any] = {"blockers": _dedupe(blockers), "dry_run": dry_run}
    if static_report_ref:
        metadata["static_report_ref"] = static_report_ref
    if plausibility_gate:
        metadata["plausibility_gate"] = plausibility_gate
    return VerificationRun(
        wb_ref=stable_ref("ver", hypothesis_ref, *metadata["blockers"]),
        hypothesis_ref=hypothesis_ref,
        prediction_lock_ref=prediction_lock_ref,
        status="blocked",
        evidence_posture="blocked",
        metrics={},
        metadata=metadata,
        parents=[hypothesis_ref, *([str(prediction_lock_ref)] if prediction_lock_ref else [])],
    )


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped
