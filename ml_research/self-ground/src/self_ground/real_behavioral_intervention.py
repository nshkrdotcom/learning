from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from self_ground.baselines import build_feature_sets
from self_ground.behavioral_tasks import (
    BehavioralTask,
    generate_behavioral_tasks,
    read_behavioral_tasks_jsonl,
    write_behavioral_tasks_jsonl,
)
from self_ground.intervention_telemetry import (
    mean_telemetry,
    telemetry_has_nonfinite,
    telemetry_warnings,
)
from self_ground.io import write_config, write_jsonl
from self_ground.logit_scoring import score_behavioral_task_logits
from self_ground.mechanism_report import build_mechanism_evidence_report
from self_ground.sae_compat import SAECompatibilityResult, verify_sae_compatibility
from self_ground.sae_interventions import run_sae_decoded_intervention_logits_with_telemetry
from self_ground.task_validation import (
    BehavioralTaskValidationSummary,
    TokenValidationResult,
    validate_behavioral_tasks,
)


@dataclass(frozen=True)
class BehavioralInterventionRun:
    out_dir: Path
    n_tasks_total: int
    n_tasks_valid: int
    n_tasks_excluded: int
    n_features_per_set: int
    n_feature_sets: int
    operations: list[str]
    patch_mode: str
    compatible: bool
    task_validation_passed: bool
    n_rows: int
    report_written: bool


BASELINE_SUMMARY_COLUMNS = [
    "family",
    "n_tasks",
    "prompt_contrast_mean",
    "prompt_contrast_abs_mean",
    "control_contrast_mean",
    "control_contrast_abs_mean",
    "intended_direction_pass_rate",
]

BEHAVIORAL_SUMMARY_COLUMNS = [
    "feature_set_label",
    "feature_selection_method",
    "operation",
    "factor",
    "patch_mode",
    "family",
    "n_tasks",
    "target_signed_delta_mean",
    "target_signed_delta_abs_mean",
    "target_absolute_delta_mean",
    "control_signed_delta_mean",
    "control_signed_delta_abs_mean",
    "control_absolute_delta_mean",
    "specificity_gap_mean",
    "collateral_ratio_mean",
    "n_null_collateral_ratio",
    "baseline_contrast_mean",
    "patched_contrast_mean",
    "control_baseline_contrast_mean",
    "control_patched_contrast_mean",
    "target_score_delta_mean",
    "foil_score_delta_mean",
    "relative_norm_drift_mean",
    "decoded_delta_norm_mean",
    "norm_drift_warning_rate",
]


def _empty_skipped_accounting() -> dict[str, Any]:
    return {
        "n_skipped_rows": 0,
        "reason_counts": {},
        "examples": [],
    }


def _record_skipped(
    accounting: dict[str, Any],
    *,
    reason: str,
    task_id: str,
    feature_set_label: str,
    operation: str,
    factor: float | None,
) -> None:
    accounting["n_skipped_rows"] += 1
    accounting["reason_counts"][reason] = accounting["reason_counts"].get(reason, 0) + 1
    if len(accounting["examples"]) < 20:
        accounting["examples"].append(
            {
                "reason": reason,
                "task_id": task_id,
                "feature_set_label": feature_set_label,
                "operation": operation,
                "factor": factor,
            }
        )


def parse_operations(value: str | list[str] | None) -> list[Literal["ablate", "amplify"]]:
    if value is None:
        return ["ablate"]
    raw = value if isinstance(value, list) else value.split(",")
    operations = [item.strip() for item in raw if item.strip()]
    if not operations or any(item not in {"ablate", "amplify"} for item in operations):
        raise ValueError("operations must be a comma-separated subset of ablate,amplify")
    return operations  # type: ignore[return-value]


def parse_amplify_factors(value: str | list[float] | None) -> list[float]:
    if value is None or value == "":
        return [2.0]
    factors = [float(item) for item in value.split(",")] if isinstance(value, str) else value
    if any(factor <= 0 for factor in factors):
        raise ValueError("amplify factors must be positive")
    if any(factor == 1.0 for factor in factors):
        raise ValueError("amplify factor 1.0 is a no-op and is rejected")
    return list(factors)


def parse_int_list(value: str | list[int] | None) -> list[int]:
    if value is None:
        return [7, 11, 13]
    if isinstance(value, list):
        return value
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _load_tasks(tasks_path: str | Path | None, per_family: int, seed: int) -> list[BehavioralTask]:
    if tasks_path is not None:
        return read_behavioral_tasks_jsonl(tasks_path)
    return generate_behavioral_tasks(per_family=per_family, seed=seed)


def _write_validation_artifacts(
    *,
    out_dir: Path,
    results: list[TokenValidationResult],
    summary: BehavioralTaskValidationSummary,
) -> None:
    write_config(
        {
            "summary": summary.model_dump(mode="json"),
            "results": [result.model_dump(mode="json") for result in results],
        },
        out_dir / "behavioral_task_validation.json",
    )
    write_jsonl(
        [result for result in results if not result.valid],
        out_dir / "excluded_behavioral_tasks.jsonl",
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _intended_pass(task: BehavioralTask, contrast: float) -> bool:
    if task.expected_baseline_direction == "positive":
        return contrast > 0
    if task.expected_baseline_direction == "negative":
        return contrast < 0
    return True


def write_baseline_task_artifacts(
    *,
    out_dir: Path,
    tasks: list[BehavioralTask],
    validations: list[TokenValidationResult],
    baseline_rows: list[dict[str, Any]],
) -> None:
    del tasks, validations
    write_jsonl(baseline_rows, out_dir / "baseline_task_scores.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in baseline_rows:
        grouped[str(row["family"])].append(row)
    with (out_dir / "baseline_task_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASELINE_SUMMARY_COLUMNS)
        writer.writeheader()
        for family in sorted(grouped):
            rows = grouped[family]
            writer.writerow(
                {
                    "family": family,
                    "n_tasks": len(rows),
                    "prompt_contrast_mean": _mean(
                        [float(row["baseline_prompt_contrast"]) for row in rows]
                    ),
                    "prompt_contrast_abs_mean": _mean(
                        [abs(float(row["baseline_prompt_contrast"])) for row in rows]
                    ),
                    "control_contrast_mean": _mean(
                        [float(row["baseline_control_contrast"]) for row in rows]
                    ),
                    "control_contrast_abs_mean": _mean(
                        [abs(float(row["baseline_control_contrast"])) for row in rows]
                    ),
                    "intended_direction_pass_rate": _mean(
                        [1.0 if row["intended_direction_pass"] else 0.0 for row in rows]
                    ),
                }
            )


def _baseline_scores(
    *,
    model_adapter,
    tasks: list[BehavioralTask],
    validations: list[TokenValidationResult],
    reduction: Literal["mean", "max"],
) -> dict[str, dict[str, Any]]:
    validation_by_id = {result.task_id: result for result in validations if result.valid}
    rows: dict[str, dict[str, Any]] = {}
    for task in tasks:
        validation = validation_by_id[task.id]
        prompt_logits = model_adapter.logits_for_texts([task.prompt])
        control_logits = model_adapter.logits_for_texts([task.control_prompt])
        score = score_behavioral_task_logits(
            task=task,
            validation=validation,
            prompt_logits=prompt_logits,
            control_logits=control_logits,
            reduction=reduction,
        )
        rows[task.id] = {
            "task_id": task.id,
            "family": task.family,
            "baseline_prompt_target_score": score.prompt_result.target_score,
            "baseline_prompt_foil_score": score.prompt_result.foil_score,
            "baseline_prompt_contrast": score.prompt_result.contrast,
            "baseline_control_target_score": score.control_result.target_score,
            "baseline_control_foil_score": score.control_result.foil_score,
            "baseline_control_contrast": score.control_result.contrast,
            "intended_direction_pass": _intended_pass(task, score.prompt_result.contrast),
        }
    return rows


def _factor_values(
    operations: list[str],
    amplify_factors: list[float],
) -> list[tuple[str, float | None]]:
    values: list[tuple[str, float | None]] = []
    for operation in operations:
        if operation == "ablate":
            values.append((operation, None))
        else:
            values.extend((operation, factor) for factor in amplify_factors)
    return values


def _finite_row(row: dict[str, Any]) -> bool:
    def finite_value(value: Any) -> bool:
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, dict):
            return all(finite_value(child) for child in value.values())
        if isinstance(value, list):
            return all(finite_value(child) for child in value)
        return True

    return finite_value(row)


def _collateral_ratio(control_abs: float, target_abs: float) -> float | None:
    if target_abs == 0:
        return None
    return control_abs / target_abs


def _result_rows(
    *,
    model_adapter,
    sae_adapter,
    tasks: list[BehavioralTask],
    validations: list[TokenValidationResult],
    baseline_rows: dict[str, dict[str, Any]],
    feature_sets: dict[str, Any],
    model_name: str,
    hook_point: str,
    sae_release: str,
    sae_id: str,
    ranking_dir: Path,
    operations: list[str],
    amplify_factors: list[float],
    patch_mode: Literal["replace", "delta"],
    token_position: int | None,
    reduction: Literal["mean", "max"],
    max_relative_norm_drift_warning: float,
    max_decoded_delta_norm_ratio_warning: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validation_by_id = {result.task_id: result for result in validations if result.valid}
    rows: list[dict[str, Any]] = []
    skipped = _empty_skipped_accounting()
    for feature_set in feature_sets["feature_sets"]:
        feature_ids = list(feature_set["feature_ids"])
        for operation, factor in _factor_values(operations, amplify_factors):
            effective_factor = 1.0 if factor is None else factor
            for task in tasks:
                validation = validation_by_id[task.id]
                (
                    prompt_logits,
                    prompt_telemetry,
                ) = run_sae_decoded_intervention_logits_with_telemetry(
                    model_adapter,
                    sae_adapter,
                    [task.prompt],
                    hook_point,
                    feature_ids,
                    operation=operation,  # type: ignore[arg-type]
                    factor=effective_factor,
                    token_position=token_position,
                    patch_mode=patch_mode,
                )
                (
                    control_logits,
                    control_telemetry,
                ) = run_sae_decoded_intervention_logits_with_telemetry(
                    model_adapter,
                    sae_adapter,
                    [task.control_prompt],
                    hook_point,
                    feature_ids,
                    operation=operation,  # type: ignore[arg-type]
                    factor=effective_factor,
                    token_position=token_position,
                    patch_mode=patch_mode,
                )
                prompt_score = score_behavioral_task_logits(
                    task=task,
                    validation=validation,
                    prompt_logits=prompt_logits,
                    control_logits=control_logits,
                    reduction=reduction,
                )
                baseline = baseline_rows[task.id]
                target_delta = (
                    prompt_score.prompt_result.contrast - baseline["baseline_prompt_contrast"]
                )
                control_delta = (
                    prompt_score.control_result.contrast - baseline["baseline_control_contrast"]
                )
                target_abs = abs(target_delta)
                control_abs = abs(control_delta)
                target_telemetry = prompt_telemetry.model_dump()
                control_telemetry_data = control_telemetry.model_dump()
                telemetry_data = mean_telemetry(target_telemetry, control_telemetry_data)
                target_warnings = telemetry_warnings(
                    target_telemetry,
                    max_relative_norm_drift_warning=max_relative_norm_drift_warning,
                    max_decoded_delta_norm_ratio_warning=max_decoded_delta_norm_ratio_warning,
                )
                control_warnings = telemetry_warnings(
                    control_telemetry_data,
                    max_relative_norm_drift_warning=max_relative_norm_drift_warning,
                    max_decoded_delta_norm_ratio_warning=max_decoded_delta_norm_ratio_warning,
                )
                warnings = {
                    "norm_drift_warning": (
                        target_warnings["norm_drift_warning"]
                        or control_warnings["norm_drift_warning"]
                    ),
                    "decoded_delta_norm_ratio_warning": (
                        target_warnings["decoded_delta_norm_ratio_warning"]
                        or control_warnings["decoded_delta_norm_ratio_warning"]
                    ),
                }
                row = {
                    "model_name": model_name,
                    "hook_point": hook_point,
                    "sae_release": sae_release,
                    "sae_id": sae_id,
                    "ranking_dir": str(ranking_dir),
                    "task_validation_status": "valid",
                    "task_id": task.id,
                    "family": task.family,
                    "feature_set_label": feature_set["label"],
                    "feature_selection_method": feature_set["selection_method"],
                    "feature_ids": feature_ids,
                    "operation": operation,
                    "factor": factor,
                    "patch_mode": patch_mode,
                    "prompt": task.prompt,
                    "target_tokens": task.target_tokens,
                    "foil_tokens": task.foil_tokens,
                    "target_token_ids": validation.target_token_ids,
                    "foil_token_ids": validation.foil_token_ids,
                    "baseline_target_score": baseline["baseline_prompt_target_score"],
                    "baseline_foil_score": baseline["baseline_prompt_foil_score"],
                    "baseline_contrast": baseline["baseline_prompt_contrast"],
                    "patched_target_score": prompt_score.prompt_result.target_score,
                    "patched_foil_score": prompt_score.prompt_result.foil_score,
                    "patched_contrast": prompt_score.prompt_result.contrast,
                    "target_signed_delta": target_delta,
                    "target_absolute_delta": target_abs,
                    "control_prompt": task.control_prompt,
                    "control_type": task.control_type,
                    "control_target_tokens": task.control_target_tokens,
                    "control_foil_tokens": task.control_foil_tokens,
                    "control_target_token_ids": validation.control_target_token_ids,
                    "control_foil_token_ids": validation.control_foil_token_ids,
                    "control_baseline_target_score": baseline[
                        "baseline_control_target_score"
                    ],
                    "control_baseline_foil_score": baseline["baseline_control_foil_score"],
                    "control_baseline_contrast": baseline["baseline_control_contrast"],
                    "control_patched_target_score": prompt_score.control_result.target_score,
                    "control_patched_foil_score": prompt_score.control_result.foil_score,
                    "control_patched_contrast": prompt_score.control_result.contrast,
                    "control_signed_delta": control_delta,
                    "control_absolute_delta": control_abs,
                    "specificity_gap": target_abs - control_abs,
                    "collateral_ratio": _collateral_ratio(control_abs, target_abs),
                    "target_intervention_telemetry": target_telemetry,
                    "control_intervention_telemetry": control_telemetry_data,
                    "telemetry_provenance": "separate_target_and_control_interventions",
                    **telemetry_data,
                    **warnings,
                    "metadata": {},
                }
                if (
                    telemetry_has_nonfinite(target_telemetry)
                    or telemetry_has_nonfinite(control_telemetry_data)
                    or telemetry_has_nonfinite(telemetry_data)
                ):
                    _record_skipped(
                        skipped,
                        reason="nonfinite_telemetry",
                        task_id=task.id,
                        feature_set_label=str(feature_set["label"]),
                        operation=operation,
                        factor=factor,
                    )
                    continue
                if not _finite_row(row):
                    _record_skipped(
                        skipped,
                        reason="nonfinite_row_value",
                        task_id=task.id,
                        feature_set_label=str(feature_set["label"]),
                        operation=operation,
                        factor=factor,
                    )
                    continue
                rows.append(row)
    return rows, skipped


def _summary_row(key: tuple, rows: list[dict[str, Any]]) -> dict[str, Any]:
    collateral_values = [
        row["collateral_ratio"] for row in rows if row["collateral_ratio"] is not None
    ]
    return {
        "feature_set_label": key[0],
        "feature_selection_method": key[1],
        "operation": key[2],
        "factor": "" if key[3] is None else key[3],
        "patch_mode": key[4],
        "family": key[5],
        "n_tasks": len(rows),
        "target_signed_delta_mean": _mean([row["target_signed_delta"] for row in rows]),
        "target_signed_delta_abs_mean": _mean([abs(row["target_signed_delta"]) for row in rows]),
        "target_absolute_delta_mean": _mean([row["target_absolute_delta"] for row in rows]),
        "control_signed_delta_mean": _mean([row["control_signed_delta"] for row in rows]),
        "control_signed_delta_abs_mean": _mean([abs(row["control_signed_delta"]) for row in rows]),
        "control_absolute_delta_mean": _mean([row["control_absolute_delta"] for row in rows]),
        "specificity_gap_mean": _mean([row["specificity_gap"] for row in rows]),
        "collateral_ratio_mean": _mean(collateral_values) if collateral_values else "",
        "n_null_collateral_ratio": len(rows) - len(collateral_values),
        "baseline_contrast_mean": _mean([row["baseline_contrast"] for row in rows]),
        "patched_contrast_mean": _mean([row["patched_contrast"] for row in rows]),
        "control_baseline_contrast_mean": _mean(
            [row["control_baseline_contrast"] for row in rows]
        ),
        "control_patched_contrast_mean": _mean(
            [row["control_patched_contrast"] for row in rows]
        ),
        "target_score_delta_mean": _mean(
            [row["patched_target_score"] - row["baseline_target_score"] for row in rows]
        ),
        "foil_score_delta_mean": _mean(
            [row["patched_foil_score"] - row["baseline_foil_score"] for row in rows]
        ),
        "relative_norm_drift_mean": _mean([row["relative_norm_drift_mean"] for row in rows]),
        "decoded_delta_norm_mean": _mean([row["decoded_delta_norm_mean"] for row in rows]),
        "norm_drift_warning_rate": _mean(
            [1.0 if row["norm_drift_warning"] else 0.0 for row in rows]
        ),
    }


def _write_behavioral_summary(rows: list[dict[str, Any]], path: Path) -> None:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        base_key = (
            row["feature_set_label"],
            row["feature_selection_method"],
            row["operation"],
            row["factor"],
            row["patch_mode"],
        )
        grouped[(*base_key, row["family"])].append(row)
        grouped[(*base_key, "__all__")].append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BEHAVIORAL_SUMMARY_COLUMNS)
        writer.writeheader()
        for key in sorted(grouped):
            writer.writerow(_summary_row(key, grouped[key]))


def _write_readme(
    *,
    out_dir: Path,
    config: dict[str, Any],
    compatibility: SAECompatibilityResult | None,
    validation_summary: BehavioralTaskValidationSummary,
    feature_sets: dict[str, Any] | None,
    claim_status: str | None,
    blocker: str | None,
    skipped_rows: dict[str, Any] | None = None,
) -> None:
    compatibility_text = "not reached"
    if compatibility is not None:
        compatibility_text = (
            f"compatible={compatibility.compatible}, "
            f"metadata={compatibility.metadata_compatible}, "
            f"shape={compatibility.shape_compatible}, "
            f"reconstruction={compatibility.reconstruction_compatible}"
        )
    blocker_text = f"\n## Blocker\n\n{blocker}\n" if blocker else ""
    diagnostic = (
        "\nThis run used --allow-metadata-mismatch and is diagnostic-only. It cannot "
        "support candidate_evidence or strong_candidate_evidence.\n"
        if config.get("allow_metadata_mismatch")
        else ""
    )
    feature_text = ""
    if feature_sets:
        feature_text = "\n".join(
            f"- {row['label']}: {', '.join(row['feature_ids'])}"
            for row in feature_sets["feature_sets"]
        )
    skipped_text = ""
    if skipped_rows is not None:
        skipped_text = f"""
## Skipped Rows

- skipped intervention rows: `{skipped_rows.get("n_skipped_rows", 0)}`
- skipped reason counts: `{skipped_rows.get("reason_counts", {})}`

"""
    text = f"""# Phase 3 Token-Contrast Evaluation

- model: `{config['model_name']}`
- hook point: `{config['hook_point']}`
- SAE release: `{config['sae_release']}`
- SAE id: `{config['sae_id']}`
- ranking dir: `{config['ranking_dir']}`
- baseline mode: `{config['baseline_mode']}`
- random seeds: `{config['random_seeds']}`
- operations: `{config['operations']}`
- patch mode: `{config['patch_mode']}`
- compatibility: `{compatibility_text}`
- valid tasks: `{validation_summary.valid_tasks}`
- excluded tasks: `{validation_summary.excluded_tasks}`
- claim status: `{claim_status}`
{diagnostic}
{blocker_text}
{skipped_text}
## Feature Sets

{feature_text}

## Interpretation

This is a real decoded SAE intervention token-contrast evaluation. It scores
target prompts and matched non-negation control prompts. It is not broad
behavioral understanding, complete mechanism discovery, model introspection, or
monosemantic feature discovery.
"""
    if blocker and "compatibility" in blocker:
        text += "\nNo behavioral intervention rows were written because SAE compatibility failed.\n"
    if blocker and "task validation" in blocker:
        text += "\nNo behavioral intervention rows were written because task validation failed.\n"
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def run_real_behavioral_sae_intervention(
    *,
    out_dir: str | Path,
    ranking_dir: str | Path,
    tasks_path: str | Path | None = None,
    per_family: int = 10,
    seed: int = 7,
    model_name: str = "EleutherAI/pythia-70m-deduped",
    hook_point: str = "blocks.2.hook_resid_post",
    sae_release: str,
    sae_id: str,
    top_k_features: int = 5,
    baseline_mode: Literal[
        "top",
        "top-vs-random",
        "top-vs-random-multiseed",
        "top-vs-bottom-active",
        "top-vs-random-and-bottom-active",
    ] = "top-vs-random-multiseed",
    random_seeds: list[int] | None = None,
    operations: list[Literal["ablate", "amplify"]] | None = None,
    amplify_factors: list[float] | None = None,
    patch_mode: Literal["replace", "delta"] = "delta",
    token_position: int | None = -1,
    device: str | None = "cpu",
    reduction: Literal["mean", "max"] = "mean",
    min_valid_tasks_per_family: int = 2,
    allow_metadata_mismatch: bool = False,
    write_report: bool = True,
    max_relative_norm_drift_warning: float = 0.5,
    max_decoded_delta_norm_ratio_warning: float = 0.5,
    model_adapter=None,
    sae_adapter=None,
) -> BehavioralInterventionRun:
    ranking_path = Path(ranking_dir)
    ranking_file = ranking_path / "feature_rankings.csv"
    if not ranking_file.exists():
        raise ValueError(
            "ranking_dir must contain feature_rankings.csv. Run SAE ranking first with "
            "scripts/run_real_activation_ranking.py --feature-source sae ..."
        )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    parsed_operations = operations or ["ablate"]
    factors = amplify_factors or [2.0]
    seeds = random_seeds or [7, 11, 13]
    if "amplify" in parsed_operations:
        parse_amplify_factors(factors)
    config = {
        "model_name": model_name,
        "hook_point": hook_point,
        "sae_release": sae_release,
        "sae_id": sae_id,
        "ranking_dir": str(ranking_path),
        "tasks_path": str(tasks_path) if tasks_path else None,
        "per_family": per_family,
        "seed": seed,
        "top_k_features": top_k_features,
        "baseline_mode": baseline_mode,
        "random_seeds": seeds,
        "operations": parsed_operations,
        "amplify_factors": factors,
        "patch_mode": patch_mode,
        "token_position": token_position,
        "device": device,
        "reduction": reduction,
        "min_valid_tasks_per_family": min_valid_tasks_per_family,
        "allow_metadata_mismatch": allow_metadata_mismatch,
        "max_relative_norm_drift_warning": max_relative_norm_drift_warning,
        "max_decoded_delta_norm_ratio_warning": max_decoded_delta_norm_ratio_warning,
    }
    write_config(config, out_path / "config.json")
    tasks = _load_tasks(tasks_path, per_family, seed)
    write_behavioral_tasks_jsonl(tasks, out_path / "behavioral_tasks.jsonl")

    if model_adapter is None:
        from self_ground.model import TransformerLensModelAdapter

        model_adapter = TransformerLensModelAdapter(model_name=model_name, device=device)

    valid_tasks, validation_results, validation_summary = validate_behavioral_tasks(
        model_adapter=model_adapter,
        tasks=tasks,
        min_valid_tasks_per_family=min_valid_tasks_per_family,
    )
    _write_validation_artifacts(
        out_dir=out_path,
        results=validation_results,
        summary=validation_summary,
    )
    if not validation_summary.passes_minimum:
        if write_report:
            build_mechanism_evidence_report(
                behavioral_run_dir=out_path,
                out_json=out_path / "mechanism_report.json",
                out_md=out_path / "mechanism_report.md",
            )
        _write_readme(
            out_dir=out_path,
            config=config,
            compatibility=None,
            validation_summary=validation_summary,
            feature_sets=None,
            claim_status="blocked" if write_report else None,
            blocker="task validation failed",
            skipped_rows=None,
        )
        return BehavioralInterventionRun(
            out_dir=out_path,
            n_tasks_total=len(tasks),
            n_tasks_valid=len(valid_tasks),
            n_tasks_excluded=validation_summary.excluded_tasks,
            n_features_per_set=top_k_features,
            n_feature_sets=0,
            operations=parsed_operations,
            patch_mode=patch_mode,
            compatible=False,
            task_validation_passed=False,
            n_rows=0,
            report_written=write_report,
        )

    if sae_adapter is None:
        from self_ground.sae import SAELensAdapter

        sae_adapter = SAELensAdapter.from_pretrained(
            release=sae_release,
            sae_id=sae_id,
            device=device or getattr(model_adapter, "device", "cpu"),
        )
    compatibility = verify_sae_compatibility(
        model_name=model_name,
        hook_point=hook_point,
        sae_release=sae_release,
        sae_id=sae_id,
        device=device,
        model_adapter=model_adapter,
        sae_adapter=sae_adapter,
        allow_metadata_mismatch=allow_metadata_mismatch,
    )
    write_config(compatibility.model_dump(), out_path / "compatibility.json")
    may_run_diagnostic = (
        allow_metadata_mismatch
        and compatibility.shape_compatible
        and compatibility.reconstruction_compatible
    )
    if not compatibility.compatible and not may_run_diagnostic:
        if write_report:
            build_mechanism_evidence_report(
                behavioral_run_dir=out_path,
                out_json=out_path / "mechanism_report.json",
                out_md=out_path / "mechanism_report.md",
            )
        _write_readme(
            out_dir=out_path,
            config=config,
            compatibility=compatibility,
            validation_summary=validation_summary,
            feature_sets=None,
            claim_status="blocked" if write_report else None,
            blocker=f"SAE compatibility failed: {compatibility.error}",
            skipped_rows=None,
        )
        return BehavioralInterventionRun(
            out_dir=out_path,
            n_tasks_total=len(tasks),
            n_tasks_valid=len(valid_tasks),
            n_tasks_excluded=validation_summary.excluded_tasks,
            n_features_per_set=top_k_features,
            n_feature_sets=0,
            operations=parsed_operations,
            patch_mode=patch_mode,
            compatible=False,
            task_validation_passed=True,
            n_rows=0,
            report_written=write_report,
        )

    feature_sets = build_feature_sets(
        ranking_file,
        top_k=top_k_features,
        baseline_mode=baseline_mode,
        random_seeds=seeds,
    )
    write_config(feature_sets, out_path / "feature_sets.json")
    baseline_rows = _baseline_scores(
        model_adapter=model_adapter,
        tasks=valid_tasks,
        validations=validation_results,
        reduction=reduction,
    )
    write_baseline_task_artifacts(
        out_dir=out_path,
        tasks=valid_tasks,
        validations=[result for result in validation_results if result.valid],
        baseline_rows=list(baseline_rows.values()),
    )
    rows, skipped_rows = _result_rows(
        model_adapter=model_adapter,
        sae_adapter=sae_adapter,
        tasks=valid_tasks,
        validations=validation_results,
        baseline_rows=baseline_rows,
        feature_sets=feature_sets,
        model_name=model_name,
        hook_point=hook_point,
        sae_release=sae_release,
        sae_id=sae_id,
        ranking_dir=ranking_path,
        operations=parsed_operations,
        amplify_factors=factors,
        patch_mode=patch_mode,
        token_position=token_position,
        reduction=reduction,
        max_relative_norm_drift_warning=max_relative_norm_drift_warning,
        max_decoded_delta_norm_ratio_warning=max_decoded_delta_norm_ratio_warning,
    )
    write_config(skipped_rows, out_path / "skipped_behavioral_rows.json")
    write_jsonl(rows, out_path / "behavioral_intervention_results.jsonl")
    _write_behavioral_summary(rows, out_path / "behavioral_summary.csv")
    claim_status = None
    if write_report:
        report = build_mechanism_evidence_report(
            behavioral_run_dir=out_path,
            out_json=out_path / "mechanism_report.json",
            out_md=out_path / "mechanism_report.md",
        )
        claim_status = report.claim_status
    _write_readme(
        out_dir=out_path,
        config=config,
        compatibility=compatibility,
        validation_summary=validation_summary,
        feature_sets=feature_sets,
        claim_status=claim_status,
        blocker=None,
        skipped_rows=skipped_rows,
    )
    return BehavioralInterventionRun(
        out_dir=out_path,
        n_tasks_total=len(tasks),
        n_tasks_valid=len(valid_tasks),
        n_tasks_excluded=validation_summary.excluded_tasks,
        n_features_per_set=top_k_features,
        n_feature_sets=len(feature_sets["feature_sets"]),
        operations=parsed_operations,
        patch_mode=patch_mode,
        compatible=compatibility.compatible,
        task_validation_passed=True,
        n_rows=len(rows),
        report_written=write_report,
    )
