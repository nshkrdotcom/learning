from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from local_mi_lab.activations import resolve_position_index
from local_mi_lab.patching import component_hook_name, patching_effect
from local_mi_lab.plots import (
    plot_controlled_patching_by_family,
    plot_controlled_patching_candidate_gap,
)
from local_mi_lab.prompts import read_prompts_csv
from local_mi_lab.tokens import token_id_for_single_token
from local_mi_lab.types import PromptRecord

DEFAULT_CONTROLLED_PATCHING_FAMILIES = [
    "positive_repeat_sequence",
    "distractor_repeat_control",
    "random_expected_token_control",
    "same_token_frequency_control",
]

RESULT_COLUMNS = [
    "example_id",
    "family",
    "control_family",
    "should_show_induction_behavior",
    "candidate_id",
    "candidate_source",
    "layer",
    "head",
    "component",
    "position_label",
    "patch_position",
    "head_specific_patch",
    "actual_patch_scope",
    "clean_prompt",
    "corrupt_prompt",
    "target_token",
    "true_expected_next_token",
    "wrong_or_control_token",
    "metric",
    "clean_score",
    "corrupt_score",
    "patched_score",
    "effect_size",
    "effect_size_status",
    "positive_minus_control_context",
]


def select_examples_for_controlled_patching(
    records: list[PromptRecord],
    families: list[str],
    examples_per_family: int,
    seed: int = 0,
) -> list[PromptRecord]:
    rng = random.Random(seed)
    selected: list[PromptRecord] = []
    for family in families:
        family_records = [record for record in records if record.family == family]
        family_records = sorted(family_records, key=lambda record: record.family_index or 0)
        rng.shuffle(family_records)
        selected.extend(sorted(family_records[:examples_per_family], key=lambda r: r.family_index or 0))
    return selected


def build_patching_jobs(
    records: list[PromptRecord],
    candidates: list[dict[str, Any]],
    families: list[str],
    examples_per_family: int,
    max_candidates: int,
    seed: int = 0,
    component_override: str | None = None,
    position_label: str = "final",
) -> list[dict[str, Any]]:
    by_id = {record.example_id: record for record in records}
    selected_records = select_examples_for_controlled_patching(
        records,
        families=families,
        examples_per_family=examples_per_family,
        seed=seed,
    )
    jobs: list[dict[str, Any]] = []
    for candidate in candidates[:max_candidates]:
        for record in selected_records:
            positive = by_id.get(record.paired_positive_example_id)
            if positive is None:
                raise ValueError(
                    f"Record {record.example_id} references missing paired positive "
                    f"{record.paired_positive_example_id!r}"
                )
            clean_prompt, corrupt_prompt, context = _clean_corrupt_for_record(record, positive)
            component = component_override or str(candidate["component"])
            jobs.append(
                {
                    "record": record,
                    "candidate": candidate,
                    "clean_prompt": clean_prompt,
                    "corrupt_prompt": corrupt_prompt,
                    "target_token": record.expected_next_token,
                    "true_expected_next_token": record.true_expected_next_token,
                    "wrong_or_control_token": (
                        record.wrong_or_control_token
                        or (
                            record.expected_next_token
                            if record.expected_next_token != record.true_expected_next_token
                            else ""
                        )
                    ),
                    "component": component,
                    "position_label": position_label,
                    "positive_minus_control_context": context,
                }
            )
    return jobs


def run_controlled_patching(
    model: Any,
    run_dir: str | Path,
    candidates_path: str | Path,
    *,
    families: list[str] | None = None,
    examples_per_family: int = 8,
    max_candidates: int = 12,
    seed: int = 0,
    component_override: str | None = None,
    position_label: str = "final",
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(run_dir)
    records = read_prompts_csv(root / "prompts.csv")
    candidates = pd.read_csv(candidates_path).to_dict(orient="records")
    selected_families = families or DEFAULT_CONTROLLED_PATCHING_FAMILIES
    jobs = build_patching_jobs(
        records,
        candidates,
        families=selected_families,
        examples_per_family=examples_per_family,
        max_candidates=max_candidates,
        seed=seed,
        component_override=component_override,
        position_label=position_label,
    )
    if dry_run:
        return {"n_jobs": len(jobs), "families": selected_families, "dry_run": True}

    rows = [_run_job(model, job) for job in tqdm(jobs, desc="Controlled patching")]
    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    by_family = aggregate_controlled_patching_by_family(results)
    by_candidate = aggregate_controlled_patching_by_candidate(results, candidates)
    summary = controlled_patching_summary(results, by_family, by_candidate, selected_families)
    write_controlled_patching_artifacts(root, results, by_family, by_candidate, summary)
    return summary


def aggregate_controlled_patching_by_family(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    valid = results.copy()
    valid["effect_size_numeric"] = pd.to_numeric(valid["effect_size"], errors="coerce")
    grouped = valid.groupby(["family", "candidate_id", "layer", "head", "component"], as_index=False)
    return grouped.agg(
        n_examples=("example_id", "nunique"),
        mean_effect_size=("effect_size_numeric", "mean"),
        median_effect_size=("effect_size_numeric", "median"),
        n_denominator_zero=(
            "effect_size_status",
            lambda values: int((values == "denominator_zero").sum()),
        ),
        fraction_positive_effect=(
            "effect_size_numeric",
            lambda values: float((values.dropna() > 0).mean()) if len(values.dropna()) else 0.0,
        ),
        should_show_induction_behavior=("should_show_induction_behavior", "first"),
    )


def aggregate_controlled_patching_by_candidate(
    results: pd.DataFrame,
    candidates: list[dict[str, Any]],
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    family = aggregate_controlled_patching_by_family(results)
    rows: list[dict[str, Any]] = []
    candidate_by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    for candidate_id, subset in family.groupby("candidate_id"):
        positive = subset[subset["should_show_induction_behavior"]]
        controls = subset[~subset["should_show_induction_behavior"]]
        positive_mean = _safe_float(positive["mean_effect_size"].mean()) if not positive.empty else None
        max_control_row = (
            controls.sort_values("mean_effect_size", ascending=False).iloc[0]
            if not controls.empty and controls["mean_effect_size"].notna().any()
            else None
        )
        max_control_mean = (
            _safe_float(max_control_row["mean_effect_size"]) if max_control_row is not None else None
        )
        gap = (
            positive_mean - max_control_mean
            if positive_mean is not None and max_control_mean is not None
            else None
        )
        total = results[results["candidate_id"] == candidate_id]
        status = classify_candidate_specificity(total, positive_mean, max_control_mean, gap)
        candidate = candidate_by_id.get(str(candidate_id), {})
        rows.append(
            {
                "candidate_id": candidate_id,
                "source": candidate.get("source", ""),
                "layer": int(subset["layer"].iloc[0]),
                "head": subset["head"].iloc[0],
                "component": subset["component"].iloc[0],
                "positive_mean_effect_size": positive_mean,
                "max_control_mean_effect_size": max_control_mean,
                "positive_minus_control_effect_gap": gap,
                "hardest_control_family": (
                    str(max_control_row["family"]) if max_control_row is not None else ""
                ),
                "n_positive_examples": int(positive["n_examples"].sum()) if not positive.empty else 0,
                "n_control_examples": int(controls["n_examples"].sum()) if not controls.empty else 0,
                "specificity_status": status,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "positive_minus_control_effect_gap",
        ascending=False,
        na_position="last",
    )


def classify_candidate_specificity(
    candidate_results: pd.DataFrame,
    positive_mean: float | None,
    max_control_mean: float | None,
    gap: float | None,
) -> str:
    if candidate_results.empty:
        return "insufficient_examples"
    denominator_zero = int((candidate_results["effect_size_status"] == "denominator_zero").sum())
    if denominator_zero >= max(1, len(candidate_results) // 2):
        return "denominator_problem"
    if positive_mean is None or max_control_mean is None:
        return "insufficient_examples"
    if positive_mean <= 0:
        return "no_positive_effect"
    if max_control_mean >= positive_mean:
        return "nonspecific_moves_controls"
    if gap is not None and gap > 0:
        return "positive_specific_candidate"
    return "insufficient_examples"


def controlled_patching_summary(
    results: pd.DataFrame,
    by_family: pd.DataFrame,
    by_candidate: pd.DataFrame,
    families: list[str],
) -> dict[str, Any]:
    positive = by_family[by_family["should_show_induction_behavior"]] if not by_family.empty else pd.DataFrame()
    controls = by_family[~by_family["should_show_induction_behavior"]] if not by_family.empty else pd.DataFrame()
    positive_mean = _safe_float(positive["mean_effect_size"].mean()) if not positive.empty else None
    max_control = _safe_float(controls["mean_effect_size"].max()) if not controls.empty else None
    best_gap = (
        _safe_float(by_candidate["positive_minus_control_effect_gap"].max())
        if not by_candidate.empty
        else None
    )
    actual_scopes = (
        sorted(str(scope) for scope in results["actual_patch_scope"].dropna().unique())
        if not results.empty
        else []
    )
    actual_scope = actual_scopes[0] if len(actual_scopes) == 1 else "mixed_patch_scopes"
    return {
        "n_result_rows": int(len(results)),
        "families": families,
        "n_candidates": int(results["candidate_id"].nunique()) if not results.empty else 0,
        "head_specific_patch": False,
        "actual_patch_scope": actual_scope,
        "actual_patch_scopes": actual_scopes,
        "positive_mean_effect_size": positive_mean,
        "max_control_mean_effect_size": max_control,
        "best_positive_minus_control_effect_gap": best_gap,
        "specificity_status_counts": (
            by_candidate["specificity_status"].value_counts().to_dict()
            if not by_candidate.empty
            else {}
        ),
        "interpretation_note": (
            "Controlled patching asks whether causal effects separate positives from controls. "
            "This is tiny causal practice, not a mechanism claim."
        ),
    }


def write_controlled_patching_artifacts(
    run_dir: Path,
    results: pd.DataFrame,
    by_family: pd.DataFrame,
    by_candidate: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    results.to_csv(run_dir / "controlled_patching_results.csv", index=False)
    by_family.to_csv(run_dir / "controlled_patching_by_family.csv", index=False)
    by_candidate.to_csv(run_dir / "controlled_patching_by_candidate.csv", index=False)
    (run_dir / "controlled_patching_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "controlled_patching_notes.md").write_text(_notes(summary), encoding="utf-8")
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    plot_controlled_patching_by_family(
        by_family,
        figures_dir / "controlled_patching_by_family.png",
    )
    plot_controlled_patching_candidate_gap(
        by_candidate,
        figures_dir / "controlled_patching_candidate_gap.png",
    )


def _run_job(model: Any, job: dict[str, Any]) -> dict[str, Any]:
    record: PromptRecord = job["record"]
    candidate = job["candidate"]
    layer = int(candidate["layer"])
    head = "" if pd.isna(candidate.get("head")) else int(candidate["head"])
    component = str(job["component"])
    hook_name = component_hook_name(component, layer)
    target_token = str(job["target_token"])
    target_id = token_id_for_single_token(model.tokenizer, target_token)
    clean_tokens = model.to_tokens(job["clean_prompt"])
    corrupt_tokens = model.to_tokens(job["corrupt_prompt"])
    clean_position = resolve_position_index(job["position_label"], int(clean_tokens.shape[1]))
    corrupt_position = resolve_position_index(job["position_label"], int(corrupt_tokens.shape[1]))
    with torch.inference_mode():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=[hook_name])
        corrupt_logits = model(corrupt_tokens)
    clean_score = float(clean_logits[0, -1, target_id].detach().cpu())
    corrupt_score = float(corrupt_logits[0, -1, target_id].detach().cpu())
    clean_acts = clean_cache[hook_name].detach()

    def patch_hook(corrupt_act: torch.Tensor, hook: Any) -> torch.Tensor:
        del hook
        patched = corrupt_act.clone()
        patched[:, corrupt_position, :] = clean_acts[:, clean_position, :]
        return patched

    with torch.inference_mode():
        patched_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=[(hook_name, patch_hook)])
    patched_score = float(patched_logits[0, -1, target_id].detach().cpu())
    effect = patching_effect(clean_score, corrupt_score, patched_score)
    return {
        "example_id": record.example_id,
        "family": record.family,
        "control_family": record.control_family,
        "should_show_induction_behavior": record.should_show_induction_behavior,
        "candidate_id": candidate["candidate_id"],
        "candidate_source": candidate["source"],
        "layer": layer,
        "head": head,
        "component": component,
        "position_label": job["position_label"],
        "patch_position": corrupt_position,
        "head_specific_patch": False,
        "actual_patch_scope": _actual_patch_scope(component),
        "clean_prompt": job["clean_prompt"],
        "corrupt_prompt": job["corrupt_prompt"],
        "target_token": target_token,
        "true_expected_next_token": job["true_expected_next_token"],
        "wrong_or_control_token": job["wrong_or_control_token"],
        "metric": "target_logit",
        "clean_score": clean_score,
        "corrupt_score": corrupt_score,
        "patched_score": patched_score,
        **effect,
        "positive_minus_control_context": job["positive_minus_control_context"],
    }


def _clean_corrupt_for_record(record: PromptRecord, positive: PromptRecord) -> tuple[str, str, str]:
    if record.family == "positive_repeat_sequence":
        return positive.prompt, positive.control_prompt, "positive_vs_builtin_control_prompt"
    if record.family == "random_expected_token_control":
        return positive.prompt, record.prompt, "same_prompt_wrong_target_token"
    return positive.prompt, record.prompt, f"positive_vs_{record.family}"


def _actual_patch_scope(component: str) -> str:
    if component == "attn_out":
        return "full_attn_out_layer"
    if component == "resid_post":
        return "full_resid_post_layer"
    return component


def _safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Controlled Patching Notes",
            "",
            "Controlled patching asks whether causal effects separate positives from controls.",
            "Layer-level attn_out patching is not head-specific.",
            "",
            f"Candidates patched: {summary['n_candidates']}",
            f"Positive mean effect size: {summary['positive_mean_effect_size']}",
            f"Max control mean effect size: {summary['max_control_mean_effect_size']}",
            f"Best positive-minus-control effect gap: {summary['best_positive_minus_control_effect_gap']}",
            f"Specificity statuses: {summary['specificity_status_counts']}",
            "",
            "This is tiny causal practice, not induction-head discovery.",
        ]
    ) + "\n"
