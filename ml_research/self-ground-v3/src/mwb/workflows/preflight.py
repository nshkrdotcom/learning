from __future__ import annotations

from typing import Any

from mwb.domain.objects import Hypothesis, PreflightReport
from mwb.refs import stable_ref
from mwb.static_compiler import StaticCompiler, has_static_compiler_payload


def _hypothesis_from_payload(payload: dict[str, Any]) -> Hypothesis:
    normalized = {
        "wb_ref": payload["wb_ref"],
        "title": payload["title"],
        "units": payload.get("units", []),
        "example_bundle_ref": payload.get("example_bundle_ref", ""),
        "control_bundle_ref": payload.get("control_bundle_ref", ""),
        "expected_effect": payload.get("expected_effect", ""),
        "required_controls": payload.get("required_controls", []),
        "alternative_explanations": payload.get("alternative_explanations", []),
        "metadata": payload.get("metadata", {}),
    }
    return Hypothesis.model_validate(normalized)


def run_preflight(payload: dict[str, Any]) -> PreflightReport:
    hypothesis = _hypothesis_from_payload(payload)
    metadata = hypothesis.metadata
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    warnings: list[str] = []

    if hypothesis.control_bundle_ref:
        checks.append({"name": "control_bundle_present", "status": "pass"})
    else:
        checks.append({"name": "control_bundle_present", "status": "fail"})
        blockers.append("missing_control_bundle")

    tensor_ok = bool(metadata.get("tensor_space_compatible", True))
    checks.append({"name": "tensor_space_compatible", "status": "pass" if tensor_ok else "fail"})
    if not tensor_ok:
        blockers.append("tensor_space_mismatch")

    hook_ok = bool(metadata.get("model_sae_hook_match", True))
    checks.append({"name": "model_sae_hook_match", "status": "pass" if hook_ok else "fail"})
    if not hook_ok:
        blockers.append("metadata_mismatch")

    if has_static_compiler_payload(payload):
        static_report = StaticCompiler().compile_payload(payload)
        checks.extend(static_report.checks)
        blockers.extend(static_report.blockers)
        warnings.extend(static_report.warnings)
    elif "decoder_unembed_projection_score" in metadata:
        score = float(metadata["decoder_unembed_projection_score"])
        if score >= 0.03:
            projection_status = "pass"
        elif score > 0.0:
            projection_status = "warn"
            warnings.append("weak_decoder_unembed_projection")
        else:
            projection_status = "fail"
            blockers.append("preflight_failed")
        checks.append(
            {
                "name": "decoder_unembed_projection",
                "status": projection_status,
                "score": score,
                "normalization": "l2_cosine",
            }
        )

    status = "fail" if blockers else "warn" if warnings else "pass"
    return PreflightReport(
        wb_ref=stable_ref("pre", hypothesis.wb_ref, status, checks),
        hypothesis_ref=hypothesis.wb_ref,
        status=status,
        checks=checks,
        blockers=blockers,
        warnings=warnings,
        parents=[hypothesis.wb_ref],
    )
