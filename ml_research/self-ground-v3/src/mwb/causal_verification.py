from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from mwb.domain.objects import (
    InterventionReceipt,
    PredictionLock,
    TelemetryReport,
    VerificationRun,
)
from mwb.project import Project
from mwb.refs import stable_ref
from mwb.sqlite_index import initialize_schema, insert_payload
from mwb.time import utc_now
from mwb.workflows.verify import run_verify

JsonDict = dict[str, Any]
EPSILON = 1e-12
DEFAULT_TELEMETRY_THRESHOLDS = {
    "kl_drift": 0.1,
    "norm_drift": 0.5,
}


class CausalVerificationService:
    def __init__(self, project: Project) -> None:
        self.project = project

    def verify_payload(
        self,
        hypothesis_payload: dict[str, Any],
        *,
        prediction_lock: PredictionLock | dict[str, Any] | None,
        diagnostic_only: bool,
        dry_run: bool,
    ) -> VerificationRun:
        gate_run = run_verify(
            hypothesis_payload,
            prediction_lock=prediction_lock,
            diagnostic_only=diagnostic_only,
            dry_run=dry_run,
        )
        if gate_run.status == "blocked":
            return self._write_blocked_run(gate_run)

        verification = hypothesis_payload.get("verification")
        if not isinstance(verification, dict):
            return self._write_minimal_run(gate_run, hypothesis_payload, dry_run=dry_run)

        hypothesis_ref = str(hypothesis_payload["wb_ref"])
        operations = list(verification.get("operations", []))
        baseline = _baseline(verification)
        thresholds = {
            **DEFAULT_TELEMETRY_THRESHOLDS,
            **verification.get("telemetry_thresholds", {}),
        }
        run_ref = stable_ref(
            "run",
            "verify",
            hypothesis_ref,
            [operation.get("operation") for operation in operations],
            "dry_run" if dry_run else gate_run.evidence_posture,
        )
        run_dir = self.project.mechanism_dir / "runs" / run_ref
        run_dir.mkdir(parents=True, exist_ok=True)

        receipts: list[InterventionReceipt] = []
        results: list[dict[str, Any]] = []
        telemetry_reports: list[TelemetryReport] = []
        blockers: list[str] = []
        policy = verification.get("policy", {})

        for index, operation in enumerate(operations):
            receipt, result, telemetry = self._execute_operation(
                run_ref=run_ref,
                hypothesis_ref=hypothesis_ref,
                operation=operation,
                operation_index=index,
                baseline=baseline,
                thresholds=thresholds,
                dry_run=dry_run,
            )
            receipts.append(receipt)
            results.append(result)
            telemetry_reports.append(telemetry)
            if telemetry.status == "fail":
                blockers.append("off_manifold_intervention")
            if (
                not diagnostic_only
                and operation.get("operation") == "zero_ablate"
                and policy.get("zero_ablation_claim_ceiling", "diagnostic_only")
                == "diagnostic_only"
            ):
                blockers.append("zero_ablation_claim_ceiling")

        blockers = _dedupe([*gate_run.metadata.get("blockers", []), *blockers])
        claim_ceiling = None
        evidence_posture = gate_run.evidence_posture
        status = "dry_run" if dry_run else "candidate_evidence"
        if diagnostic_only:
            evidence_posture = "diagnostic_only"
            status = "diagnostic_only" if not dry_run else "dry_run"
        if "zero_ablation_claim_ceiling" in blockers:
            evidence_posture = "diagnostic_only"
            status = "diagnostic_only"
            claim_ceiling = "diagnostic_only"
        elif blockers:
            status = "insufficient_evidence"

        metrics = _aggregate_metrics(results, telemetry_reports)
        metadata = {
            **gate_run.metadata,
            "run_ref": run_ref,
            "run_dir": str(run_dir),
            "operation_count": len(operations),
            "blockers": blockers,
        }
        if claim_ceiling:
            metadata["claim_ceiling"] = claim_ceiling
        run = VerificationRun(
            wb_ref=stable_ref("ver", run_ref, status, evidence_posture, metrics, blockers),
            hypothesis_ref=hypothesis_ref,
            prediction_lock_ref=gate_run.prediction_lock_ref,
            status=status,
            evidence_posture=evidence_posture,
            metrics=metrics,
            metadata=metadata,
            parents=gate_run.parents,
        )
        self._write_artifacts(
            run=run,
            run_dir=run_dir,
            receipts=receipts,
            results=results,
            telemetry_reports=telemetry_reports,
            blockers=blockers,
            claim_bearing=evidence_posture == "claim_bearing",
        )
        return run

    def resample_ablate_sae_feature(
        self,
        *,
        hypothesis_ref: str,
        model: Any,
        sae: Any,
        hook_point: str,
        feature_index: int,
        clean_prompt: str,
        corrupt_prompt: str,
        target_token: str,
        foil_token: str,
        diagnostic_only: bool,
    ) -> VerificationRun:
        import torch

        clean_tokens = model.to_tokens(clean_prompt)
        corrupt_tokens = model.to_tokens(corrupt_prompt)
        target_token_id = int(model.to_single_token(target_token))
        foil_token_id = int(model.to_single_token(foil_token))
        with torch.no_grad():
            clean_logits, clean_cache = model.run_with_cache(
                clean_tokens,
                names_filter=lambda name: name == hook_point,
            )
            _, corrupt_cache = model.run_with_cache(
                corrupt_tokens,
                names_filter=lambda name: name == hook_point,
            )
            clean_activation = clean_cache[hook_point]
            corrupt_activation = corrupt_cache[hook_point]
            clean_features = sae.encode(clean_activation)
            corrupt_features = sae.encode(corrupt_activation)
            patched_features = clean_features.clone()
            patched_features[..., feature_index] = corrupt_features[..., feature_index]
            clean_reconstruction = sae.decode(clean_features)
            patched_reconstruction = sae.decode(patched_features)
            patched_activation = clean_activation + (patched_reconstruction - clean_reconstruction)

            def patch_hook(value, hook):
                return patched_activation.to(value.device)

            patched_logits = model.run_with_hooks(
                clean_tokens,
                fwd_hooks=[(hook_point, patch_hook)],
            )

        baseline_target = float(clean_logits[0, -1, target_token_id])
        baseline_control = float(clean_logits[0, -1, foil_token_id])
        patched_target = float(patched_logits[0, -1, target_token_id])
        patched_control = float(patched_logits[0, -1, foil_token_id])
        payload = {
            "wb_ref": hypothesis_ref,
            "title": "real adapter resample ablation",
            "units": [f"feature_{feature_index}"],
            "example_bundle_ref": "real_adapter_bundle",
            "control_bundle_ref": "real_adapter_controls",
            "expected_effect": "target_delta > controls",
            "required_controls": [],
            "verification": {
                "baseline": {
                    "target_logit": baseline_target,
                    "control_logit": baseline_control,
                    "logits": [
                        baseline_target,
                        baseline_control,
                    ],
                    "activation_norm": float(clean_activation.float().norm()),
                },
                "operations": [
                    {
                        "operation": "resample_ablate",
                        "unit_ref": f"feature_{feature_index}",
                        "patch_mode": "resample",
                        "patch_source": "corrupt_prompt",
                        "patch_target": "clean_prompt",
                        "coefficient": 1.0,
                        "intervened": {
                            "target_logit": patched_target,
                            "control_logit": patched_control,
                            "logits": [patched_target, patched_control],
                            "activation_norm": float(patched_activation.float().norm()),
                        },
                    }
                ],
            },
        }
        return self.verify_payload(
            payload,
            prediction_lock=None,
            diagnostic_only=diagnostic_only,
            dry_run=False,
        )

    def _execute_operation(
        self,
        *,
        run_ref: str,
        hypothesis_ref: str,
        operation: dict[str, Any],
        operation_index: int,
        baseline: dict[str, Any],
        thresholds: dict[str, float],
        dry_run: bool,
    ) -> tuple[InterventionReceipt, dict[str, Any], TelemetryReport]:
        operation_name = str(operation["operation"])
        intervened = operation.get("intervened", {})
        metric_results = _metric_results(baseline, intervened)
        receipt_ref = stable_ref("receipt", run_ref, operation_index, operation_name, operation)
        telemetry = _telemetry_report(
            run_ref=run_ref,
            receipt_ref=receipt_ref,
            operation_name=operation_name,
            baseline=baseline,
            intervened=intervened,
            thresholds=thresholds,
        )
        receipt = InterventionReceipt(
            wb_ref=receipt_ref,
            run_ref=run_ref,
            hypothesis_ref=hypothesis_ref,
            operation=operation_name,
            unit_ref=operation.get("unit_ref"),
            patch_mode=str(operation.get("patch_mode") or operation_name),
            patch_source=operation.get("patch_source"),
            patch_target=operation.get("patch_target"),
            coefficient=float(operation.get("coefficient", 1.0)),
            backend_executed=not dry_run,
            causal_direction=_causal_direction(operation_name),
            metric_results=metric_results,
            telemetry_ref=telemetry.wb_ref,
            parents=[hypothesis_ref],
        )
        result = {
            "wb_ref": stable_ref("result", receipt_ref, metric_results),
            "run_ref": run_ref,
            "hypothesis_ref": hypothesis_ref,
            "receipt_ref": receipt_ref,
            "operation": operation_name,
            "status": "dry_run" if dry_run else "executed",
            "evidence_posture": "diagnostic_only",
            "claim_bearing": False,
            "metric_results": metric_results,
            "blockers": ["off_manifold_intervention"] if telemetry.status == "fail" else [],
            "created_at": utc_now(),
        }
        return receipt, result, telemetry

    def _write_blocked_run(self, run: VerificationRun) -> VerificationRun:
        run_ref = stable_ref("run", "verify", run.hypothesis_ref, run.status, run.metadata)
        run_dir = self.project.mechanism_dir / "runs" / run_ref
        run_dir.mkdir(parents=True, exist_ok=True)
        updated = run.model_copy(
            update={
                "metadata": {
                    **run.metadata,
                    "run_ref": run_ref,
                    "run_dir": str(run_dir),
                }
            }
        )
        self._write_artifacts(
            run=updated,
            run_dir=run_dir,
            receipts=[],
            results=[],
            telemetry_reports=[],
            blockers=list(updated.metadata.get("blockers", [])),
            claim_bearing=False,
        )
        return updated

    def _write_minimal_run(
        self,
        run: VerificationRun,
        hypothesis_payload: dict[str, Any],
        *,
        dry_run: bool,
    ) -> VerificationRun:
        hypothesis_ref = str(hypothesis_payload["wb_ref"])
        run_ref = stable_ref("run", "verify", hypothesis_ref, "minimal", run.status)
        run_dir = self.project.mechanism_dir / "runs" / run_ref
        run_dir.mkdir(parents=True, exist_ok=True)
        blockers = ["artifact_incomplete"] if dry_run else []
        updated = run.model_copy(
            update={
                "metadata": {
                    **run.metadata,
                    "run_ref": run_ref,
                    "run_dir": str(run_dir),
                    "operation_count": 0,
                    "blockers": _dedupe([*run.metadata.get("blockers", []), *blockers]),
                }
            }
        )
        self._write_artifacts(
            run=updated,
            run_dir=run_dir,
            receipts=[],
            results=[],
            telemetry_reports=[],
            blockers=list(updated.metadata.get("blockers", [])),
            claim_bearing=False,
        )
        return updated

    def _write_artifacts(
        self,
        *,
        run: VerificationRun,
        run_dir: Path,
        receipts: list[InterventionReceipt],
        results: list[dict[str, Any]],
        telemetry_reports: list[TelemetryReport],
        blockers: list[str],
        claim_bearing: bool,
    ) -> None:
        manifest = {
            "run_ref": run.metadata["run_ref"],
            "source_kind": "mwb_verify",
            "source_hypothesis_ref": run.hypothesis_ref,
            "status": run.status,
            "claim_bearing": claim_bearing,
            "evidence_posture": run.evidence_posture,
            "created_at": utc_now(),
        }
        _write_json(run_dir / "run_manifest.json", manifest)
        _write_json(run_dir / "verification_run.json", run.model_dump(mode="json"))
        _write_json(run_dir / "control_metrics.json", run.metrics)
        _write_jsonl(
            run_dir / "intervention_receipts.jsonl",
            [receipt.model_dump(mode="json") for receipt in receipts],
        )
        _write_jsonl(run_dir / "verification_results.jsonl", results)
        _write_jsonl(
            run_dir / "telemetry.jsonl",
            [telemetry.model_dump(mode="json") for telemetry in telemetry_reports],
        )
        if blockers:
            _write_json(
                run_dir / "blocker_report.json",
                {
                    "wb_ref": stable_ref("blocker", run.metadata["run_ref"], blockers),
                    "wb_type": "BlockerReport",
                    "run_ref": run.metadata["run_ref"],
                    "blockers": blockers,
                    "primary_blocker": blockers[0],
                    "blocking_metrics": [],
                    "parents": [run.metadata["run_ref"]],
                },
            )
        initialize_schema(self.project.sqlite_path)
        insert_payload(
            self.project.sqlite_path,
            "verification_runs",
            run.wb_ref,
            run.model_dump(mode="json"),
        )
        for receipt in receipts:
            insert_payload(
                self.project.sqlite_path,
                "intervention_receipts",
                receipt.wb_ref,
                receipt.model_dump(mode="json"),
            )
        for result in results:
            insert_payload(
                self.project.sqlite_path,
                "verification_results",
                result["wb_ref"],
                result,
            )
        for telemetry in telemetry_reports:
            insert_payload(
                self.project.sqlite_path,
                "telemetry_reports",
                telemetry.wb_ref,
                telemetry.model_dump(mode="json"),
            )


def _baseline(verification: dict[str, Any]) -> dict[str, Any]:
    baseline = verification.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("verification.baseline is required")
    return baseline


def _metric_results(baseline: dict[str, Any], intervened: dict[str, Any]) -> dict[str, float]:
    baseline_target = float(baseline["target_logit"])
    baseline_control = float(baseline["control_logit"])
    intervened_target = float(intervened["target_logit"])
    intervened_control = float(intervened["control_logit"])
    target_delta = baseline_target - intervened_target
    control_delta = baseline_control - intervened_control
    baseline_margin = baseline_target - baseline_control
    intervened_margin = intervened_target - intervened_control
    return {
        "baseline_margin": baseline_margin,
        "intervened_margin": intervened_margin,
        "target_delta": target_delta,
        "matched_control_delta": control_delta,
        "specificity_gap": target_delta - control_delta,
        "effect_size": baseline_margin - intervened_margin,
    }


def _telemetry_report(
    *,
    run_ref: str,
    receipt_ref: str,
    operation_name: str,
    baseline: dict[str, Any],
    intervened: dict[str, Any],
    thresholds: dict[str, float],
) -> TelemetryReport:
    kl_drift = _kl_divergence(
        _softmax([float(value) for value in baseline.get("logits", [])]),
        _softmax([float(value) for value in intervened.get("logits", [])]),
    )
    baseline_norm = float(baseline.get("activation_norm", 0.0))
    intervened_norm = float(intervened.get("activation_norm", baseline_norm))
    norm_drift = abs(intervened_norm - baseline_norm) / max(abs(baseline_norm), EPSILON)
    status = (
        "fail"
        if kl_drift > float(thresholds["kl_drift"])
        or norm_drift > float(thresholds["norm_drift"])
        else "pass"
    )
    return TelemetryReport(
        wb_ref=stable_ref("telemetry", receipt_ref, kl_drift, norm_drift, status),
        run_ref=run_ref,
        receipt_ref=receipt_ref,
        operation=operation_name,
        kl_drift=kl_drift,
        activation_norm_drift=norm_drift,
        status=status,
        thresholds=dict(thresholds),
    )


def _aggregate_metrics(
    results: list[dict[str, Any]],
    telemetry_reports: list[TelemetryReport],
) -> dict[str, Any]:
    if not results:
        return {}
    target_deltas = [
        float(result["metric_results"]["target_delta"])
        for result in results
    ]
    control_deltas = [
        float(result["metric_results"]["matched_control_delta"])
        for result in results
    ]
    return {
        "operation_count": len(results),
        "target_delta": sum(target_deltas) / len(target_deltas),
        "matched_control_delta": sum(control_deltas) / len(control_deltas),
        "specificity_gap": sum(
            float(result["metric_results"]["specificity_gap"]) for result in results
        )
        / len(results),
        "kl_drift": max((telemetry.kl_drift for telemetry in telemetry_reports), default=0.0),
        "activation_norm_drift": max(
            (telemetry.activation_norm_drift for telemetry in telemetry_reports),
            default=0.0,
        ),
    }


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return [1.0]
    offset = max(values)
    exp_values = [math.exp(value - offset) for value in values]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def _kl_divergence(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("logit dimensions do not match")
    return sum(
        left_value * math.log(max(left_value, EPSILON) / max(right_value, EPSILON))
        for left_value, right_value in zip(left, right, strict=True)
    )


def _causal_direction(operation_name: str) -> str | None:
    if operation_name == "noising":
        return "clean_to_corrupt"
    if operation_name == "denoising":
        return "corrupt_to_clean"
    return None


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _write_json(path: Path, payload: JsonDict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
