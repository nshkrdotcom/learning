from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from local_mi_lab.config import write_config
from local_mi_lab.head_hooks import HeadPatchSite, resolve_head_patch_site
from local_mi_lab.head_patching import (
    _run_head_job,
    clean_corrupt_prompts,
    compute_mean_ablation_activations,
)
from local_mi_lab.heldout_prompts import generate_heldout_induction_prompts
from local_mi_lab.paths import make_run_dir, resolve_repo_path
from local_mi_lab.prompts import write_prompts_csv
from local_mi_lab.types import PromptRecord

DEFAULT_HELDOUT_FAMILIES = [
    "heldout_symbolic_longer",
    "heldout_word_sequences",
    "heldout_number_sequences",
    "heldout_double_repeat",
    "heldout_wrong_target_same_prompt",
    "heldout_no_structure_same_tokens",
]

DEFAULT_INTERVENTIONS = [
    "head_clean_to_corrupt_patch",
    "head_zero_ablation",
    "head_mean_ablation",
]

DEFAULT_POSITIONS = ["final", "previous_occurrence"]

RESULT_COLUMNS = [
    "seed",
    "candidate_id",
    "candidate_group",
    "layer",
    "head",
    "family",
    "example_id",
    "heldout_family_type",
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


def run_heldout_robustness(
    model: Any,
    config: dict[str, Any],
    candidate_set: str | Path,
    *,
    families: list[str] | None = None,
    interventions: list[str] | None = None,
    positions: list[str] | None = None,
    examples_per_family: int = 12,
    metric: str = "true_vs_control_logit_diff",
) -> Path:
    seed = int(config["experiment"].get("seed", 0))
    run_dir = make_run_dir(config["outputs"]["run_root"], config["experiment"]["name"])
    write_config(config, run_dir / "config.yaml")
    candidates = read_candidate_set(candidate_set)
    candidates.to_csv(run_dir / "candidate_set.csv", index=False)
    records = generate_heldout_induction_prompts(
        n_examples_per_family=int(config["task"]["n_examples_initial_per_family"]),
        families=list(config["task"]["families"]),
        seed=seed,
    )
    write_prompts_csv(records, run_dir / "prompts.csv")
    selected_families = families or DEFAULT_HELDOUT_FAMILIES
    selected_interventions = interventions or DEFAULT_INTERVENTIONS
    selected_positions = positions or DEFAULT_POSITIONS
    jobs = expand_heldout_jobs(
        records,
        candidates,
        families=selected_families,
        interventions=selected_interventions,
        positions=selected_positions,
        examples_per_family=examples_per_family,
        seed=seed,
    )
    site_by_layer = {
        int(layer): HeadPatchSite(**resolve_head_patch_site(model, int(layer)))
        for layer in sorted({int(job["layer"]) for job in jobs})
    }
    rows: list[dict[str, Any]] = []
    for intervention in selected_interventions:
        intervention_jobs = [job for job in jobs if job["intervention"] == intervention]
        if intervention == "head_mean_ablation":
            mean_acts = compute_mean_ablation_activations(model, intervention_jobs, site_by_layer)
            for job in intervention_jobs:
                key = job.get("_mean_ablation_key")
                if key is not None:
                    job["mean_act"], job["mean_ablation_source_n_examples"] = mean_acts.get(
                        key,
                        (None, 0),
                    )
        for job in tqdm(intervention_jobs, desc=f"Held-out {intervention}"):
            site = site_by_layer[int(job["layer"])]
            row = _run_head_job(
                model,
                job,
                run_id=run_dir.name,
                seed=seed,
                site=site,
                metric=metric,
                intervention=intervention,
            )
            rows.append(_heldout_row(row, job))
    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    by_family = aggregate_heldout_by_family(results)
    by_candidate = aggregate_heldout_by_candidate(by_family)
    summary = heldout_summary(results, by_candidate, selected_families, selected_interventions, selected_positions)
    write_heldout_artifacts(run_dir, results, by_family, by_candidate, summary)
    return run_dir


def read_candidate_set(path: str | Path) -> pd.DataFrame:
    rows = pd.read_csv(resolve_repo_path(path))
    return rows[rows["include_in_main"].astype(bool)].copy()


def expand_heldout_jobs(
    records: list[PromptRecord],
    candidates: pd.DataFrame,
    *,
    families: list[str],
    interventions: list[str],
    positions: list[str],
    examples_per_family: int,
    seed: int,
) -> list[dict[str, Any]]:
    selected = selected_records_by_family(records, families, examples_per_family, seed)
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
                            "candidate_id": candidate.candidate_id,
                            "candidate_group": candidate.candidate_group,
                            "clean_prompt": clean_prompt,
                            "corrupt_prompt": corrupt_prompt,
                            "position_label": position,
                            "intervention": intervention,
                        }
                    )
    return jobs


def selected_records_by_family(
    records: list[PromptRecord],
    families: list[str],
    examples_per_family: int,
    seed: int,
) -> list[PromptRecord]:
    rng = random.Random(seed)
    selected: list[PromptRecord] = []
    for family in families:
        rows = [record for record in records if record.family == family]
        rows = sorted(rows, key=lambda record: record.family_index or 0)
        rng.shuffle(rows)
        selected.extend(sorted(rows[:examples_per_family], key=lambda record: record.family_index or 0))
    return selected


def aggregate_heldout_by_family(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    rows = results.copy()
    rows["effect_size_numeric"] = pd.to_numeric(rows["effect_size"], errors="coerce")
    grouped = rows.groupby(
        [
            "seed",
            "candidate_id",
            "candidate_group",
            "layer",
            "head",
            "family",
            "heldout_family_type",
            "intervention",
            "position_label",
            "head_specific_patch",
            "actual_patch_scope",
            "metric",
        ],
        as_index=False,
    )
    return grouped.agg(
        n_examples=("example_id", "nunique"),
        n_valid_examples=("effect_size_numeric", lambda values: int(values.notna().sum())),
        mean_effect_size=("effect_size_numeric", "mean"),
        median_effect_size=("effect_size_numeric", "median"),
        n_position_unavailable=("position_status", lambda values: int((values == "unavailable_for_family").sum())),
        n_denominator_zero=("effect_size_status", lambda values: int((values == "denominator_zero").sum())),
    )


def aggregate_heldout_by_candidate(by_family: pd.DataFrame) -> pd.DataFrame:
    if by_family.empty:
        return pd.DataFrame()
    rows = []
    group_cols = [
        "seed",
        "candidate_id",
        "candidate_group",
        "layer",
        "head",
        "intervention",
        "position_label",
        "head_specific_patch",
        "actual_patch_scope",
        "metric",
    ]
    for key, group in by_family.groupby(group_cols):
        key_dict = dict(zip(group_cols, key, strict=True))
        positives = group[group["heldout_family_type"] == "positive"]
        controls = group[group["heldout_family_type"] == "control"]
        positive_family_means = positives.dropna(subset=["mean_effect_size"])
        control_family_means = controls.dropna(subset=["mean_effect_size"])
        positive_mean = (
            float(positive_family_means["mean_effect_size"].mean())
            if not positive_family_means.empty
            else None
        )
        max_control = (
            float(control_family_means["mean_effect_size"].max())
            if not control_family_means.empty
            else None
        )
        gap = (
            positive_mean - max_control
            if positive_mean is not None and max_control is not None
            else None
        )
        n_positive_families = int((positive_family_means["mean_effect_size"] > 0).sum())
        n_control_moving = int((control_family_means["mean_effect_size"] > 0).sum())
        rows.append(
            {
                **key_dict,
                "n_families": int(group["family"].nunique()),
                "n_examples": int(group["n_examples"].sum()),
                "positive_family_mean_effect": positive_mean,
                "max_control_family_mean_effect": max_control,
                "positive_minus_control_gap": gap,
                "n_positive_families_with_gap_gt_0": n_positive_families,
                "n_control_families_moving": n_control_moving,
                "survival_status": classify_heldout_seed_status(
                    head_specific=bool(key_dict["head_specific_patch"]),
                    positive_mean=positive_mean,
                    max_control=max_control,
                    gap=gap,
                    n_positive_families_with_gap_gt_0=n_positive_families,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "positive_minus_control_gap",
        ascending=False,
        na_position="last",
    )


def classify_heldout_seed_status(
    *,
    head_specific: bool,
    positive_mean: float | None,
    max_control: float | None,
    gap: float | None,
    n_positive_families_with_gap_gt_0: int,
) -> str:
    if not head_specific:
        return "not_head_specific"
    if positive_mean is None or max_control is None or gap is None:
        return "insufficient_valid_examples"
    if positive_mean <= 0:
        return "falsified_no_positive_effect"
    if max_control >= positive_mean:
        return "falsified_controls_move"
    if gap > 0 and n_positive_families_with_gap_gt_0 >= 2:
        return "heldout_survives_seed"
    if gap > 0:
        return "downgraded_weak_family_specific"
    return "falsified_sign_flip"


def heldout_summary(
    results: pd.DataFrame,
    by_candidate: pd.DataFrame,
    families: list[str],
    interventions: list[str],
    positions: list[str],
) -> dict[str, Any]:
    return {
        "n_result_rows": int(len(results)),
        "n_candidates": int(results["candidate_id"].nunique()) if not results.empty else 0,
        "families": families,
        "interventions": interventions,
        "positions": positions,
        "metric": "true_vs_control_logit_diff",
        "head_specific_patch": bool(results["head_specific_patch"].all()) if not results.empty else False,
        "survival_status_counts": (
            by_candidate["survival_status"].value_counts().to_dict()
            if not by_candidate.empty
            else {}
        ),
        "interpretation_note": (
            "Held-out robustness compares fixed candidates against held-out controls, "
            "interventions, and positions. This is not a mechanism claim."
        ),
    }


def write_heldout_artifacts(
    run_dir: Path,
    results: pd.DataFrame,
    by_family: pd.DataFrame,
    by_candidate: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    results.to_csv(run_dir / "heldout_robustness_results.csv", index=False)
    by_family.to_csv(run_dir / "heldout_robustness_by_family.csv", index=False)
    by_candidate.to_csv(run_dir / "heldout_robustness_by_candidate.csv", index=False)
    (run_dir / "heldout_robustness_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    plot_candidate_gaps(by_candidate, figures / "heldout_candidate_gaps.png")
    plot_family_gaps(by_family, figures / "heldout_family_gaps.png")
    plot_intervention_position_grid(by_candidate, figures / "heldout_intervention_position_grid.png")


def _heldout_row(row: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    record: PromptRecord = job["record"]
    return {
        "seed": row["seed"],
        "candidate_id": job["candidate_id"],
        "candidate_group": job["candidate_group"],
        "layer": row["layer"],
        "head": row["head"],
        "family": row["family"],
        "example_id": row["example_id"],
        "heldout_family_type": record.heldout_family_type,
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


def plot_candidate_gaps(by_candidate: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if by_candidate.empty:
        ax.text(0.5, 0.5, "No candidate rows", ha="center", va="center")
        ax.axis("off")
    else:
        plot_rows = by_candidate.copy()
        plot_rows["positive_minus_control_gap"] = pd.to_numeric(
            plot_rows["positive_minus_control_gap"],
            errors="coerce",
        )
        top = plot_rows.dropna(subset=["positive_minus_control_gap"]).sort_values(
            "positive_minus_control_gap",
            ascending=False,
        ).head(24)
        if top.empty:
            ax.text(0.5, 0.5, "No finite candidate gaps", ha="center", va="center")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(path, dpi=160)
            plt.close(fig)
            return
        labels = [
            f"{row.candidate_id}\nL{int(row.layer)}H{int(row.head)}"
            for row in top.itertuples()
        ]
        ax.bar(labels, top["positive_minus_control_gap"], color="#0b6e4f")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel("Positive-minus-control effect gap")
        ax.set_title("Held-out candidate gaps")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_family_gaps(by_family: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if by_family.empty:
        ax.text(0.5, 0.5, "No family rows", ha="center", va="center")
        ax.axis("off")
    else:
        summary = by_family.groupby("family", as_index=False).agg(
            mean_effect_size=("mean_effect_size", "mean")
        )
        ax.bar(summary["family"], summary["mean_effect_size"], color="#8f2d56")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel("Mean effect size")
        ax.set_title("Held-out effect by family")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_intervention_position_grid(by_candidate: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if by_candidate.empty:
        ax.text(0.5, 0.5, "No candidate rows", ha="center", va="center")
        ax.axis("off")
    else:
        grid = by_candidate.pivot_table(
            index="intervention",
            columns="position_label",
            values="positive_minus_control_gap",
            aggfunc="mean",
        )
        image = ax.imshow(grid.fillna(0).values, aspect="auto", cmap="coolwarm")
        ax.set_xticks(range(len(grid.columns)))
        ax.set_xticklabels(grid.columns)
        ax.set_yticks(range(len(grid.index)))
        ax.set_yticklabels(grid.index)
        ax.set_title("Mean gap by intervention and position")
        fig.colorbar(image, ax=ax, label="Mean gap")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _summary_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Held-Out Robustness Summary",
            "",
            f"Candidates tested: {summary['n_candidates']}",
            f"Families: {summary['families']}",
            f"Interventions: {summary['interventions']}",
            f"Positions: {summary['positions']}",
            f"Status counts: {summary['survival_status_counts']}",
            "",
            "This is held-out local MI practice evidence, not an induction-head discovery.",
        ]
    ) + "\n"
