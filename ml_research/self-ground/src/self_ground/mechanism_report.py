from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from self_ground.io import read_json, read_jsonl, write_config


class EvidenceThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_abs_delta_mean: float = 1e-6
    min_baseline_gap_absolute: float = 1e-6
    min_task_family_count_for_candidate: int = 2
    min_task_family_count_for_strong: int = 3
    min_valid_tasks_for_candidate: int = 6
    min_valid_tasks_for_strong: int = 30
    min_random_baseline_seeds_for_strong: int = 3
    min_top_vs_control_ratio: float = 1.05
    max_collateral_ratio_for_candidate: float = 1.0
    max_collateral_ratio_for_strong: float = 0.5
    max_relative_norm_drift_for_strong: float = 0.5
    min_intended_direction_pass_rate_for_candidate: float = 0.5


class ThresholdCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    value: float | int | bool | None
    threshold: float | int | bool | None
    details: str


class FeatureSetEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_set_label: str
    feature_ids: list[str]
    activation_abs_score_mean: float | None
    target_absolute_delta_mean: float
    control_absolute_delta_mean: float | None
    specificity_gap_mean: float | None
    collateral_ratio_mean: float | None
    n_null_collateral_ratio: int
    baseline_gap_absolute: float | None
    top_vs_control_ratio: float | None
    intended_direction_pass_rate: float | None
    n_activation_pairs: int | None
    n_behavioral_tasks: int
    families: list[str]
    threshold_checks: list[ThresholdCheck]
    limitations: list[str]


class MechanismEvidenceReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    model_name: str
    hook_point: str
    sae_release: str
    sae_id: str
    metadata_mismatch_override_used: bool
    diagnostic_only: bool
    feature_sets: list[FeatureSetEvidence]
    claim_status: Literal[
        "blocked",
        "insufficient_evidence",
        "candidate_evidence",
        "strong_candidate_evidence",
    ]
    recommended_claim: str
    not_supported_claims: list[str]
    limitations: list[str]
    row_accounting: dict[str, Any]
    artifacts: dict[str, str]
    qc: dict[str, bool]
    thresholds: EvidenceThresholds


NOT_SUPPORTED = [
    "This does not establish complete negation mechanism discovery.",
    "This does not establish broad model understanding of negation.",
    "This does not establish genuine model introspection.",
    "This does not prove the selected feature set is monosemantic.",
    "This does not prove behavior outside the evaluated token-contrast tasks.",
]

REQUIRED_EVIDENCE_ARTIFACTS = [
    "config.json",
    "compatibility.json",
    "behavioral_task_validation.json",
    "feature_sets.json",
    "baseline_task_scores.jsonl",
    "behavioral_intervention_results.jsonl",
    "behavioral_summary.csv",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in {None, ""}:
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{key} is not finite: {value!r}")
    return parsed


def _artifact_paths(run_dir: Path) -> dict[str, str]:
    names = [
        *REQUIRED_EVIDENCE_ARTIFACTS,
        "skipped_behavioral_rows.json",
        "mechanism_report.json",
        "mechanism_report.md",
    ]
    return {name: str(run_dir / name) for name in names if (run_dir / name).exists()}


def _required_artifact_status(run_dir: Path) -> dict[str, bool]:
    return {name: (run_dir / name).exists() for name in REQUIRED_EVIDENCE_ARTIFACTS}


def _baseline_pass_rate(run_dir: Path) -> float | None:
    path = run_dir / "baseline_task_scores.jsonl"
    if not path.exists():
        return None
    rows = read_jsonl(path)
    if not rows:
        return None
    return sum(1 for row in rows if bool(row.get("intended_direction_pass"))) / len(rows)


def _raw_behavioral_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "behavioral_intervention_results.jsonl"
    return read_jsonl(path) if path.exists() else []


def _valid_families(raw_rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row["family"]) for row in raw_rows if row.get("family")})


def _feature_ids(feature_sets: dict[str, Any], label: str) -> list[str]:
    for row in feature_sets.get("feature_sets", []):
        if row.get("label") == label:
            return list(row.get("feature_ids", []))
    return []


def _load_row_accounting(run_dir: Path, raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    path = run_dir / "skipped_behavioral_rows.json"
    if path.exists():
        accounting = read_json(path)
    else:
        accounting = {"n_skipped_rows": 0, "reason_counts": {}, "examples": []}
    accounting["n_written_rows"] = len(raw_rows)
    accounting["all_rows_skipped"] = (
        int(accounting.get("n_skipped_rows", 0)) > 0 and len(raw_rows) == 0
    )
    return accounting


def _valid_all_family_summary_rows(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    required_numeric = [
        "n_tasks",
        "target_absolute_delta_mean",
        "control_absolute_delta_mean",
        "specificity_gap_mean",
        "baseline_contrast_mean",
        "relative_norm_drift_mean",
        "norm_drift_warning_rate",
    ]
    valid: list[dict[str, str]] = []
    reasons: dict[str, int] = {}
    for row in rows:
        try:
            for key in required_numeric:
                if row.get(key) in {None, ""}:
                    raise ValueError(f"{key} is missing")
                _float(row, key)
            valid.append(row)
        except Exception as exc:
            reason = str(exc)
            reasons[reason] = reasons.get(reason, 0) + 1
    return valid, {
        "n_summary_rows": len(rows),
        "n_valid_summary_rows": len(valid),
        "n_invalid_summary_rows": len(rows) - len(valid),
        "invalid_summary_reason_counts": reasons,
    }


def _top_vs_control_ratio(rows: list[dict[str, str]], top_row: dict[str, str]) -> float | None:
    controls = [
        row
        for row in rows
        if row.get("feature_set_label") != "top"
        and row.get("operation") == top_row.get("operation")
        and row.get("factor") == top_row.get("factor")
        and row.get("patch_mode") == top_row.get("patch_mode")
    ]
    if not controls:
        return None
    control_mean = sum(_float(row, "target_absolute_delta_mean") or 0.0 for row in controls) / len(
        controls
    )
    if control_mean <= 0:
        return None
    top_value = _float(top_row, "target_absolute_delta_mean")
    if top_value is None:
        return None
    return top_value / control_mean


def _build_feature_evidence(
    *,
    rows: list[dict[str, str]],
    feature_sets: dict[str, Any],
    thresholds: EvidenceThresholds,
    baseline_pass_rate: float | None,
    families: list[str],
) -> list[FeatureSetEvidence]:
    evidence: list[FeatureSetEvidence] = []
    for row in rows:
        label = str(row["feature_set_label"])
        target_abs = _float(row, "target_absolute_delta_mean") or 0.0
        control_abs = _float(row, "control_absolute_delta_mean")
        collateral = _float(row, "collateral_ratio_mean")
        baseline_gap = abs(_float(row, "baseline_contrast_mean") or 0.0)
        relative_norm_drift = _float(row, "relative_norm_drift_mean") or 0.0
        norm_drift_warning_rate = _float(row, "norm_drift_warning_rate") or 0.0
        ratio = _top_vs_control_ratio(rows, row) if label == "top" else None
        checks = [
            ThresholdCheck(
                name="nonzero_target_delta",
                passed=target_abs > thresholds.min_abs_delta_mean,
                value=target_abs,
                threshold=thresholds.min_abs_delta_mean,
                details="Top feature-set target movement must be finite and nonzero.",
            ),
            ThresholdCheck(
                name="baseline_calibration",
                passed=(baseline_pass_rate or 0.0)
                >= thresholds.min_intended_direction_pass_rate_for_candidate,
                value=baseline_pass_rate,
                threshold=thresholds.min_intended_direction_pass_rate_for_candidate,
                details="Baseline intended-direction pass rate.",
            ),
            ThresholdCheck(
                name="family_count",
                passed=len(families) >= thresholds.min_task_family_count_for_candidate,
                value=len(families),
                threshold=thresholds.min_task_family_count_for_candidate,
                details="At least two task families are required for candidate evidence.",
            ),
            ThresholdCheck(
                name="collateral_ratio",
                passed=collateral is not None
                and collateral <= thresholds.max_collateral_ratio_for_candidate,
                value=collateral,
                threshold=thresholds.max_collateral_ratio_for_candidate,
                details="Matched non-negation control movement must remain bounded.",
            ),
            ThresholdCheck(
                name="norm_drift_not_dominant",
                passed=norm_drift_warning_rate < 0.5,
                value=norm_drift_warning_rate,
                threshold=0.5,
                details="Norm drift warnings must not dominate evaluated rows.",
            ),
        ]
        if label == "top":
            checks.append(
                ThresholdCheck(
                    name="top_beats_control_feature_sets",
                    passed=ratio is not None and ratio >= thresholds.min_top_vs_control_ratio,
                    value=ratio,
                    threshold=thresholds.min_top_vs_control_ratio,
                    details="Top features must beat deterministic control feature sets.",
                )
            )
        limitations = [
            "Evidence is limited to deterministic next-token contrast tasks.",
            "Residual decoded SAE patching is configuration-specific.",
            "Activation-matched feature-set controls are not implemented in Phase 3.",
        ]
        if collateral is None:
            limitations.append("Collateral ratio is null because target deltas are zero.")
        if norm_drift_warning_rate > 0:
            limitations.append("Some rows exceeded configured norm-drift warning thresholds.")
        if relative_norm_drift > thresholds.max_relative_norm_drift_for_strong:
            limitations.append("Relative norm drift exceeds the strong-evidence threshold.")
        evidence.append(
            FeatureSetEvidence(
                feature_set_label=label,
                feature_ids=_feature_ids(feature_sets, label),
                activation_abs_score_mean=None,
                target_absolute_delta_mean=target_abs,
                control_absolute_delta_mean=control_abs,
                specificity_gap_mean=_float(row, "specificity_gap_mean"),
                collateral_ratio_mean=collateral,
                n_null_collateral_ratio=int(row.get("n_null_collateral_ratio") or 0),
                baseline_gap_absolute=baseline_gap,
                top_vs_control_ratio=ratio,
                intended_direction_pass_rate=baseline_pass_rate,
                n_activation_pairs=None,
                n_behavioral_tasks=int(_float(row, "n_tasks") or 0),
                families=families,
                threshold_checks=checks,
                limitations=limitations,
            )
        )
    return evidence


def _claim_status(
    *,
    compatibility: dict[str, Any] | None,
    validation: dict[str, Any] | None,
    evidence: list[FeatureSetEvidence],
    config: dict[str, Any],
    thresholds: EvidenceThresholds,
    required_artifacts: dict[str, bool],
    row_accounting: dict[str, Any],
    summary_quality: dict[str, Any],
    summary_rows: list[dict[str, str]],
) -> str:
    if not all(required_artifacts.values()):
        return "blocked"
    if summary_quality.get("n_summary_rows", 0) and not summary_rows:
        return "blocked"
    if bool(row_accounting.get("all_rows_skipped")):
        return "blocked"
    if not compatibility or not validation:
        return "blocked"
    if not bool(compatibility.get("compatible")) and not bool(compatibility.get("diagnostic_only")):
        return "blocked"
    summary = validation.get("summary", validation)
    if not bool(summary.get("passes_minimum")):
        return "blocked"
    if not evidence:
        return "blocked"
    top = next((item for item in evidence if item.feature_set_label == "top"), None)
    if top is None:
        return "insufficient_evidence"
    if bool(compatibility.get("diagnostic_only")) or bool(config.get("allow_metadata_mismatch")):
        return "insufficient_evidence"
    candidate_checks = all(check.passed for check in top.threshold_checks)
    if (
        not candidate_checks
        or top.n_behavioral_tasks < thresholds.min_valid_tasks_for_candidate
        or top.target_absolute_delta_mean <= thresholds.min_abs_delta_mean
    ):
        return "insufficient_evidence"
    actual_operations = {
        str(row.get("operation"))
        for row in summary_rows
        if row.get("feature_set_label") == "top"
    }
    random_control_count = len(
        {
            str(row.get("feature_set_label"))
            for row in summary_rows
            if str(row.get("feature_set_label", "")).startswith("random_seed_")
        }
    )
    top_rows = [row for row in summary_rows if row.get("feature_set_label") == "top"]
    max_top_norm_drift = max(
        [_float(row, "relative_norm_drift_mean") or 0.0 for row in top_rows],
        default=0.0,
    )
    max_top_warning_rate = max(
        [_float(row, "norm_drift_warning_rate") or 0.0 for row in top_rows],
        default=0.0,
    )
    if int(row_accounting.get("n_skipped_rows", 0)) > 0:
        return "insufficient_evidence"
    if (
        top.n_behavioral_tasks >= thresholds.min_valid_tasks_for_strong
        and len(top.families) >= thresholds.min_task_family_count_for_strong
        and random_control_count >= thresholds.min_random_baseline_seeds_for_strong
        and {"ablate", "amplify"} <= actual_operations
        and (top.collateral_ratio_mean or 999.0) <= thresholds.max_collateral_ratio_for_strong
        and max_top_norm_drift <= thresholds.max_relative_norm_drift_for_strong
        and max_top_warning_rate == 0.0
    ):
        return "strong_candidate_evidence"
    return "candidate_evidence"


def _top_level_limitations(
    *,
    config: dict[str, Any],
    compatibility: dict[str, Any] | None,
    baseline_pass_rate: float | None,
    evidence: list[FeatureSetEvidence],
    row_accounting: dict[str, Any],
    summary_quality: dict[str, Any],
    summary_rows: list[dict[str, str]],
    thresholds: EvidenceThresholds,
) -> list[str]:
    limitations = ["Activation-matched control feature sets are not implemented in Phase 3."]
    if compatibility and (
        compatibility.get("diagnostic_only") or config.get("allow_metadata_mismatch")
    ):
        limitations.append("Metadata mismatch override was used; run is diagnostic-only.")
    if baseline_pass_rate is not None and baseline_pass_rate < (
        thresholds.min_intended_direction_pass_rate_for_candidate
    ):
        limitations.append("Baseline intended-direction pass rate is below candidate threshold.")
    if int(row_accounting.get("n_skipped_rows", 0)) > 0:
        limitations.append("Some intervention rows were skipped due to non-finite values.")
    if summary_quality.get("n_invalid_summary_rows", 0):
        limitations.append("Some behavioral summary rows were malformed or non-finite.")
    operations = {
        str(row.get("operation"))
        for row in summary_rows
        if row.get("feature_set_label") == "top"
    }
    if "amplify" not in operations:
        limitations.append("No amplification sweep is present in the actual summary rows.")
    top = next((item for item in evidence if item.feature_set_label == "top"), None)
    if top and top.n_null_collateral_ratio:
        limitations.append("Some collateral ratios are null because target deltas are zero.")
    high_drift = any(
        (_float(row, "relative_norm_drift_mean") or 0.0)
        > thresholds.max_relative_norm_drift_for_strong
        for row in summary_rows
        if row.get("feature_set_label") == "top"
    )
    if high_drift:
        limitations.append("Top feature-set relative norm drift exceeds strong threshold.")
    return limitations


def _write_markdown(report: MechanismEvidenceReport, path: Path) -> None:
    text = f"""# Phase 3 Token-Contrast Evidence Report

- model: `{report.model_name}`
- hook point: `{report.hook_point}`
- SAE release: `{report.sae_release}`
- SAE id: `{report.sae_id}`
- diagnostic only: `{report.diagnostic_only}`
- claim status: `{report.claim_status}`

## Recommended Claim

{report.recommended_claim}

## Limitations

"""
    for limitation in report.limitations:
        text += f"- {limitation}\n"
    text += "\n## Threshold Checks\n\n"
    for feature_set in report.feature_sets:
        text += f"### {feature_set.feature_set_label}\n\n"
        for check in feature_set.threshold_checks:
            text += (
                f"- {check.name}: `{check.passed}` value=`{check.value}` "
                f"threshold=`{check.threshold}`\n"
            )
        text += "\n"
    text += "## Not Supported\n\n"
    for claim in report.not_supported_claims:
        text += f"- {claim}\n"
    text += f"""
## Row Accounting

```json
{report.row_accounting}
```

## Rerun

```bash
uv run python scripts/run_phase3_behavioral_evaluation.py \\
  --ranking-dir runs/test_real_sae_ranking \\
  --out runs/test_phase3_behavioral_evaluation \\
  --model {report.model_name} \\
  --hook-point {report.hook_point} \\
  --sae-release {report.sae_release} \\
  --sae-id {report.sae_id} \\
  --per-family 2 \\
  --top-k-features 2 \\
  --baseline-mode top-vs-random-multiseed \\
  --random-seeds 7,11,13 \\
  --operations ablate \\
  --patch-mode delta \\
  --device cpu \\
  --write-report
```
"""
    path.write_text(text, encoding="utf-8")


def build_mechanism_evidence_report(
    *,
    behavioral_run_dir: str | Path,
    ranking_dir: str | Path | None = None,
    thresholds: EvidenceThresholds | None = None,
    out_json: str | Path | None = None,
    out_md: str | Path | None = None,
) -> MechanismEvidenceReport:
    del ranking_dir
    run_dir = Path(behavioral_run_dir)
    threshold_config = thresholds or EvidenceThresholds()
    config = read_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
    compatibility = (
        read_json(run_dir / "compatibility.json")
        if (run_dir / "compatibility.json").exists()
        else None
    )
    validation = (
        read_json(run_dir / "behavioral_task_validation.json")
        if (run_dir / "behavioral_task_validation.json").exists()
        else None
    )
    feature_sets = (
        read_json(run_dir / "feature_sets.json") if (run_dir / "feature_sets.json").exists() else {}
    )
    raw_summary_rows = (
        [
            row
            for row in _read_csv(run_dir / "behavioral_summary.csv")
            if row.get("family") == "__all__"
        ]
        if (run_dir / "behavioral_summary.csv").exists()
        else []
    )
    summary_rows, summary_quality = _valid_all_family_summary_rows(raw_summary_rows)
    raw_rows = _raw_behavioral_rows(run_dir)
    row_accounting = _load_row_accounting(run_dir, raw_rows)
    required_artifacts = _required_artifact_status(run_dir)
    baseline_pass_rate = _baseline_pass_rate(run_dir)
    families = _valid_families(raw_rows)
    if not families and validation is not None:
        summary = validation.get("summary", validation)
        families = sorted(
            family
            for family, count in dict(summary.get("valid_by_family", {})).items()
            if int(count) > 0
        )
    evidence = _build_feature_evidence(
        rows=summary_rows,
        feature_sets=feature_sets,
        thresholds=threshold_config,
        baseline_pass_rate=baseline_pass_rate,
        families=families,
    )
    claim_status = _claim_status(
        compatibility=compatibility,
        validation=validation,
        evidence=evidence,
        config=config,
        thresholds=threshold_config,
        required_artifacts=required_artifacts,
        row_accounting=row_accounting,
        summary_quality=summary_quality,
        summary_rows=summary_rows,
    )
    limitations = _top_level_limitations(
        config=config,
        compatibility=compatibility,
        baseline_pass_rate=baseline_pass_rate,
        evidence=evidence,
        row_accounting=row_accounting,
        summary_quality=summary_quality,
        summary_rows=summary_rows,
        thresholds=threshold_config,
    )
    recommended_claim = (
        "The selected SAE feature set has candidate evidence for influencing "
        "negation-sensitive next-token contrasts under decoded residual intervention "
        "in this model/hook/SAE configuration."
        if claim_status in {"candidate_evidence", "strong_candidate_evidence"}
        else "The run does not support a candidate mechanism claim under the configured thresholds."
    )
    report = MechanismEvidenceReport(
        schema_version="phase3.token_contrast.v1",
        model_name=str(config.get("model_name") or config.get("model") or ""),
        hook_point=str(config.get("hook_point") or ""),
        sae_release=str(config.get("sae_release") or ""),
        sae_id=str(config.get("sae_id") or ""),
        metadata_mismatch_override_used=bool(config.get("allow_metadata_mismatch")),
        diagnostic_only=bool(
            config.get("allow_metadata_mismatch")
            or (compatibility or {}).get("diagnostic_only")
        ),
        feature_sets=evidence,
        claim_status=claim_status,  # type: ignore[arg-type]
        recommended_claim=recommended_claim,
        not_supported_claims=NOT_SUPPORTED,
        limitations=limitations,
        row_accounting={
            **row_accounting,
            **summary_quality,
        },
        artifacts=_artifact_paths(run_dir),
        qc={
            "compatibility_artifact_present": compatibility is not None,
            "task_validation_artifact_present": validation is not None,
            "behavioral_rows_present": (run_dir / "behavioral_intervention_results.jsonl").exists(),
            "required_artifacts_present": all(required_artifacts.values()),
        },
        thresholds=threshold_config,
    )
    if out_json is not None:
        write_config(report.model_dump(mode="json"), out_json)
    if out_md is not None:
        _write_markdown(report, Path(out_md))
    return report
