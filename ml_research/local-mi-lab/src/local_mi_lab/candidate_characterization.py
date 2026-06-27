from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from local_mi_lab.attention_effect_alignment import run_attention_effect_alignment
from local_mi_lab.characterization_prompts import generate_characterization_prompts
from local_mi_lab.head_circuit_diagnostics import run_head_circuit_diagnostics
from local_mi_lab.head_hooks import HeadPatchSite, resolve_head_patch_site
from local_mi_lab.head_patching import (
    _run_head_job,
    clean_corrupt_prompts,
    compute_mean_ablation_activations,
)
from local_mi_lab.paths import resolve_repo_path
from local_mi_lab.position_characterization import (
    aggregate_position_by_candidate,
    position_characterization_summary,
    write_position_characterization_artifacts,
)
from local_mi_lab.prompts import write_prompts_csv
from local_mi_lab.types import PromptRecord

DEFAULT_CHARACTERIZATION_POSITIONS = ["final", "previous_occurrence"]
DEFAULT_CHARACTERIZATION_INTERVENTIONS = [
    "head_clean_to_corrupt_patch",
    "head_zero_ablation",
    "head_mean_ablation",
]

CHARACTERIZATION_RESULT_COLUMNS = [
    "seed",
    "candidate_id",
    "candidate_group",
    "layer",
    "head",
    "family",
    "heldout_family_type",
    "example_id",
    "intervention",
    "intervention_status",
    "position_label",
    "position_status",
    "head_specific_patch",
    "actual_patch_scope",
    "metric",
    "clean_score",
    "corrupt_score",
    "patched_score",
    "effect_size",
    "effect_size_status",
    "true_expected_next_token",
    "wrong_or_control_token",
    "clean_prompt",
    "corrupt_prompt",
]


def run_candidate_characterization(
    model: Any,
    config: dict[str, Any],
    candidate_set: str | Path,
    *,
    output_dir: str | Path,
    examples_per_family: int = 4,
    interventions: list[str] | None = None,
    positions: list[str] | None = None,
    metric: str = "true_vs_control_logit_diff",
) -> dict[str, Any]:
    output_root = resolve_repo_path(output_dir)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory already exists and is non-empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    seed = int(config["experiment"].get("seed", 20))
    records = generate_characterization_prompts(
        n_examples_per_family=int(config["task"]["n_examples_initial_per_family"]),
        families=list(config["task"]["families"]),
        seed=seed,
    )
    write_prompts_csv(records, output_root / "prompts.csv")
    candidates = pd.read_csv(resolve_repo_path(candidate_set))
    candidates = candidates[candidates["include_in_main"].astype(bool)].copy()
    candidates.to_csv(output_root / "candidate_set.csv", index=False)
    selected_interventions = interventions or DEFAULT_CHARACTERIZATION_INTERVENTIONS
    selected_positions = positions or DEFAULT_CHARACTERIZATION_POSITIONS
    jobs = expand_characterization_jobs(
        records,
        candidates,
        examples_per_family=examples_per_family,
        seed=seed,
        positions=selected_positions,
        interventions=selected_interventions,
    )
    sites = {
        int(layer): HeadPatchSite(**resolve_head_patch_site(model, int(layer)))
        for layer in sorted({int(job["layer"]) for job in jobs})
    }
    rows: list[dict[str, Any]] = []
    for intervention in selected_interventions:
        intervention_jobs = [job for job in jobs if job["intervention"] == intervention]
        if intervention == "head_mean_ablation":
            mean_activations = compute_mean_ablation_activations(model, intervention_jobs, sites)
            for job in intervention_jobs:
                key = job.get("_mean_ablation_key")
                if key is not None:
                    job["mean_act"], job["mean_ablation_source_n_examples"] = mean_activations.get(
                        key,
                        (None, 0),
                    )
        for job in tqdm(intervention_jobs, desc=f"Characterization {intervention}"):
            result = _run_head_job(
                model,
                job,
                run_id=output_root.name,
                seed=seed,
                site=sites[int(job["layer"])],
                metric=metric,
                intervention=intervention,
            )
            rows.append(_characterization_result_row(result, job))
    results = pd.DataFrame(rows, columns=CHARACTERIZATION_RESULT_COLUMNS)
    results.to_csv(output_root / "candidate_characterization_results.csv", index=False)
    results.to_csv(output_root / "heldout_robustness_results.csv", index=False)
    position_dir = output_root / "position_characterization"
    position_dir.mkdir(exist_ok=True)
    position_by_candidate = aggregate_position_by_candidate(results)
    position_summary = position_characterization_summary(results, position_by_candidate)
    write_position_characterization_artifacts(
        position_dir,
        results,
        position_by_candidate,
        position_summary,
    )
    attention_summary = run_attention_effect_alignment(
        model,
        heldout_run=output_root,
        candidate_set=output_root / "candidate_set.csv",
        output_dir=output_root / "attention_effect_alignment",
    )
    diagnostic_summary = run_head_circuit_diagnostics(
        model,
        config,
        output_root / "candidate_set.csv",
        output_dir=output_root / "head_circuit_diagnostics",
        examples_per_family=examples_per_family,
    )
    by_candidate = summarize_candidate_characterization(
        results,
        pd.read_csv(output_root / "attention_effect_alignment" / "attention_effect_by_candidate.csv"),
        position_by_candidate,
        pd.read_csv(output_root / "head_circuit_diagnostics" / "head_circuit_diagnostics_by_candidate.csv"),
    )
    by_candidate.to_csv(output_root / "candidate_characterization_by_candidate.csv", index=False)
    summary = candidate_characterization_summary(
        seed=seed,
        results=results,
        by_candidate=by_candidate,
        attention_summary=attention_summary,
        position_summary=position_summary,
        diagnostic_summary=diagnostic_summary,
        interventions=selected_interventions,
        positions=selected_positions,
        examples_per_family=examples_per_family,
    )
    (output_root / "candidate_characterization_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "candidate_characterization.md").write_text(
        _candidate_characterization_markdown(summary, by_candidate),
        encoding="utf-8",
    )
    return summary


def expand_characterization_jobs(
    records: list[PromptRecord],
    candidates: pd.DataFrame,
    *,
    examples_per_family: int,
    seed: int,
    positions: list[str],
    interventions: list[str],
) -> list[dict[str, Any]]:
    selected = _selected_records(records, examples_per_family, seed)
    by_id = {record.example_id: record for record in records}
    jobs: list[dict[str, Any]] = []
    for candidate in candidates.itertuples(index=False):
        for record in selected:
            positive = by_id.get(record.paired_positive_example_id)
            if positive is None:
                raise ValueError(f"Missing paired positive {record.paired_positive_example_id!r}")
            clean_prompt, corrupt_prompt = clean_corrupt_prompts(record, positive)
            for intervention in interventions:
                for position in positions:
                    jobs.append(
                        {
                            "record": record,
                            "layer": int(candidate.layer),
                            "head": int(candidate.head),
                            "candidate_id": str(candidate.candidate_id),
                            "candidate_group": str(candidate.candidate_group),
                            "clean_prompt": clean_prompt,
                            "corrupt_prompt": corrupt_prompt,
                            "position_label": position,
                            "intervention": intervention,
                        }
                    )
    return jobs


def summarize_candidate_characterization(
    results: pd.DataFrame,
    attention_by_candidate: pd.DataFrame,
    position_by_candidate: pd.DataFrame,
    diagnostics_by_candidate: pd.DataFrame,
) -> pd.DataFrame:
    effect_summary = _effect_summary_by_candidate(results)
    merged = effect_summary.merge(
        attention_by_candidate[
            [
                "candidate_id",
                "spearman_attention_effect_corr",
                "mean_source_attention_margin",
            ]
        ],
        on="candidate_id",
        how="left",
    )
    merged = merged.merge(
        position_by_candidate[
            ["candidate_id", "position_specificity_status"]
        ],
        on="candidate_id",
        how="left",
    )
    merged = merged.merge(
        diagnostics_by_candidate[
            [
                "candidate_id",
                "ov_copy_margin",
                "ov_status",
                "qk_source_margin",
                "qk_status",
            ]
        ],
        on="candidate_id",
        how="left",
    )
    merged["characterization_seed_status"] = [
        classify_characterization_seed_status(row)
        for row in merged.to_dict(orient="records")
    ]
    return merged.sort_values(
        "positive_minus_control_gap",
        ascending=False,
        na_position="last",
    )


def classify_characterization_seed_status(row: dict[str, Any]) -> str:
    gap = _optional_float(row.get("positive_minus_control_gap"))
    positive = _optional_float(row.get("mean_true_vs_control_effect"))
    max_control = _optional_float(row.get("max_control_effect"))
    corr = _optional_float(row.get("spearman_attention_effect_corr"))
    position_status = str(row.get("position_specificity_status") or "")
    ov_status = str(row.get("ov_status") or "")
    qk_status = str(row.get("qk_status") or "")
    if gap is None or positive is None:
        return "insufficient_characterization_data"
    if max_control is not None and max_control >= positive:
        return "characterization_falsifies"
    if corr is not None and corr <= 0:
        return "characterization_falsifies"
    if position_status in {"distractor_like", "no_position_effect"}:
        return "characterization_falsifies"
    if ov_status == "ov_contradicts_copy" and qk_status == "qk_contradicts_source_selection":
        return "characterization_falsifies"
    diagnostic_support = ov_status == "ov_supports_copy" or qk_status == "qk_supports_source_selection"
    position_support = position_status in {
        "destination_specific",
        "source_specific",
        "both_source_and_destination",
        "position_nonspecific",
    }
    if gap > 0 and positive > 0 and (corr is None or corr > 0) and position_support and diagnostic_support:
        return "characterization_supports"
    if gap > 0 or positive > 0:
        return "characterization_downgrades"
    return "characterization_falsifies"


def candidate_characterization_summary(
    *,
    seed: int,
    results: pd.DataFrame,
    by_candidate: pd.DataFrame,
    attention_summary: dict[str, Any],
    position_summary: dict[str, Any],
    diagnostic_summary: dict[str, Any],
    interventions: list[str],
    positions: list[str],
    examples_per_family: int,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "n_result_rows": int(len(results)),
        "n_candidates": int(by_candidate["candidate_id"].nunique()) if not by_candidate.empty else 0,
        "examples_per_family": examples_per_family,
        "interventions": interventions,
        "positions": positions,
        "primary_metric": "true_vs_control_logit_diff",
        "characterization_status_counts": (
            by_candidate["characterization_seed_status"].value_counts().to_dict()
            if not by_candidate.empty
            else {}
        ),
        "attention_alignment_status_counts": attention_summary.get("alignment_status_counts", {}),
        "position_status_counts": position_summary.get("position_status_counts", {}),
        "ov_status_counts": diagnostic_summary.get("ov_status_counts", {}),
        "qk_status_counts": diagnostic_summary.get("qk_status_counts", {}),
        "interpretation_note": (
            "Candidate characterization combines local diagnostics. It does not establish "
            "an induction head, a full circuit, or a broad GPT-2 claim."
        ),
    }


def _effect_summary_by_candidate(results: pd.DataFrame) -> pd.DataFrame:
    rows = results.copy()
    rows["effect_size_numeric"] = pd.to_numeric(rows["effect_size"], errors="coerce")
    family_means = (
        rows.groupby(
            [
                "seed",
                "candidate_id",
                "candidate_group",
                "layer",
                "head",
                "family",
                "heldout_family_type",
            ],
            as_index=False,
        )
        .agg(mean_effect=("effect_size_numeric", "mean"))
    )
    summaries: list[dict[str, Any]] = []
    for key, group in family_means.groupby(["seed", "candidate_id", "candidate_group", "layer", "head"]):
        seed, candidate_id, candidate_group, layer, head = key
        positives = group[group["heldout_family_type"] == "positive"]["mean_effect"].dropna()
        controls = group[group["heldout_family_type"] == "control"]["mean_effect"].dropna()
        positive_mean = float(positives.mean()) if not positives.empty else None
        max_control = float(controls.max()) if not controls.empty else None
        summaries.append(
            {
                "seed": int(seed),
                "candidate_id": candidate_id,
                "candidate_group": candidate_group,
                "layer": int(layer),
                "head": int(head),
                "mean_true_vs_control_effect": positive_mean,
                "max_control_effect": max_control,
                "positive_minus_control_gap": (
                    positive_mean - max_control
                    if positive_mean is not None and max_control is not None
                    else None
                ),
            }
        )
    return pd.DataFrame(summaries)


def _selected_records(
    records: list[PromptRecord],
    examples_per_family: int,
    seed: int,
) -> list[PromptRecord]:
    selected: list[PromptRecord] = []
    rng = __import__("random").Random(seed)
    for family in sorted({record.family for record in records}):
        rows = [record for record in records if record.family == family]
        rows = sorted(rows, key=lambda record: record.family_index or 0)
        rng.shuffle(rows)
        selected.extend(sorted(rows[:examples_per_family], key=lambda record: record.family_index or 0))
    return selected


def _characterization_result_row(row: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    record: PromptRecord = job["record"]
    return {
        "seed": row["seed"],
        "candidate_id": job["candidate_id"],
        "candidate_group": job["candidate_group"],
        "layer": row["layer"],
        "head": row["head"],
        "family": row["family"],
        "heldout_family_type": record.heldout_family_type,
        "example_id": row["example_id"],
        "intervention": row["intervention"],
        "intervention_status": row["intervention_status"],
        "position_label": row["position_label"],
        "position_status": row["position_status"],
        "head_specific_patch": row["head_specific_patch"],
        "actual_patch_scope": row["actual_patch_scope"],
        "metric": row["metric"],
        "clean_score": row["clean_score"],
        "corrupt_score": row["corrupt_score"],
        "patched_score": row["patched_score"],
        "effect_size": row["effect_size"],
        "effect_size_status": row["effect_size_status"],
        "true_expected_next_token": row["true_expected_next_token"],
        "wrong_or_control_token": row["wrong_or_control_token"],
        "clean_prompt": row["clean_prompt"],
        "corrupt_prompt": row["corrupt_prompt"],
    }


def _candidate_characterization_markdown(
    summary: dict[str, Any],
    by_candidate: pd.DataFrame,
) -> str:
    lines = [
        "# Candidate Characterization Seed Summary",
        "",
        f"- Seed: `{summary['seed']}`",
        f"- Candidates: `{summary['n_candidates']}`",
        f"- Rows: `{summary['n_result_rows']}`",
        f"- Status counts: `{summary['characterization_status_counts']}`",
        "",
        "This is a local characterization artifact, not an induction-head discovery.",
        "",
        "| candidate | head | group | status | gap | attention corr | OV | QK |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    if by_candidate.empty:
        lines.append("| none |  |  |  |  |  |  |  |")
    else:
        for row in by_candidate.itertuples(index=False):
            lines.append(
                f"| {row.candidate_id} | L{int(row.layer)}H{int(row.head)} | "
                f"{row.candidate_group} | {row.characterization_seed_status} | "
                f"{_fmt(row.positive_minus_control_gap)} | "
                f"{_fmt(row.spearman_attention_effect_corr)} | "
                f"{row.ov_status} | {row.qk_status} |"
            )
    return "\n".join(lines) + "\n"


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.4f}"
