from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from local_mi_lab.head_hooks import HeadPatchSite, resolve_head_patch_site
from local_mi_lab.head_patching import _run_head_job, clean_corrupt_prompts
from local_mi_lab.heldout_prompts import generate_heldout_induction_prompts
from local_mi_lab.paths import resolve_repo_path
from local_mi_lab.types import PromptRecord

DEFAULT_POSITION_FAMILIES = [
    "heldout_symbolic_longer",
    "heldout_word_sequences",
    "heldout_number_sequences",
    "heldout_double_repeat",
    "heldout_wrong_target_same_prompt",
    "heldout_no_structure_same_tokens",
]

DEFAULT_POSITIONS = ["final", "previous_occurrence", "source_position", "distractor_position"]

RESULT_COLUMNS = [
    "seed",
    "candidate_id",
    "candidate_group",
    "layer",
    "head",
    "family",
    "heldout_family_type",
    "example_id",
    "intervention",
    "position_label",
    "position_status",
    "head_specific_patch",
    "actual_patch_scope",
    "metric",
    "effect_size",
    "effect_size_status",
    "clean_prompt",
    "corrupt_prompt",
]


def run_position_characterization(
    model: Any,
    config: dict[str, Any],
    candidate_set: str | Path,
    *,
    output_dir: str | Path,
    families: list[str] | None = None,
    examples_per_family: int = 12,
    positions: list[str] | None = None,
    intervention: str = "head_clean_to_corrupt_patch",
    metric: str = "true_vs_control_logit_diff",
) -> dict[str, Any]:
    output_root = resolve_repo_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    seed = int(config["experiment"].get("seed", 0))
    records = generate_heldout_induction_prompts(
        n_examples_per_family=int(config["task"]["n_examples_initial_per_family"]),
        families=list(config["task"]["families"]),
        seed=seed,
    )
    candidates = pd.read_csv(resolve_repo_path(candidate_set))
    candidates = candidates[candidates["include_in_main"].astype(bool)].copy()
    jobs = expand_position_jobs(
        records,
        candidates,
        families=families or DEFAULT_POSITION_FAMILIES,
        examples_per_family=examples_per_family,
        seed=seed,
        positions=positions or DEFAULT_POSITIONS,
    )
    sites = {
        int(layer): HeadPatchSite(**resolve_head_patch_site(model, int(layer)))
        for layer in sorted({int(job["layer"]) for job in jobs})
    }
    rows: list[dict[str, Any]] = []
    for job in tqdm(jobs, desc="Position characterization"):
        row = _run_head_job(
            model,
            job,
            run_id=Path(output_root).name,
            seed=seed,
            site=sites[int(job["layer"])],
            metric=metric,
            intervention=intervention,
        )
        rows.append(_position_row(row, job))
    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    by_candidate = aggregate_position_by_candidate(results)
    summary = position_characterization_summary(results, by_candidate)
    write_position_characterization_artifacts(output_root, results, by_candidate, summary)
    return summary


def expand_position_jobs(
    records: list[PromptRecord],
    candidates: pd.DataFrame,
    *,
    families: list[str],
    examples_per_family: int,
    seed: int,
    positions: list[str],
) -> list[dict[str, Any]]:
    selected = _selected_records(records, families, examples_per_family, seed)
    by_id = {record.example_id: record for record in records}
    jobs: list[dict[str, Any]] = []
    for candidate in candidates.itertuples(index=False):
        for record in selected:
            positive = by_id.get(record.paired_positive_example_id)
            if positive is None:
                raise ValueError(f"Missing paired positive {record.paired_positive_example_id!r}")
            clean_prompt, corrupt_prompt = clean_corrupt_prompts(record, positive)
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
                    }
                )
    return jobs


def aggregate_position_by_candidate(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "candidate_group",
                "layer",
                "head",
                "final_effect",
                "previous_occurrence_effect",
                "source_position_effect",
                "distractor_position_effect",
                "final_minus_source",
                "source_minus_distractor",
                "position_specificity_status",
            ]
        )
    table = results.copy()
    table["effect_size_numeric"] = pd.to_numeric(table["effect_size"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for key, group in table.groupby(["candidate_id", "candidate_group", "layer", "head"]):
        candidate_id, candidate_group, layer, head = key
        means = group.groupby("position_label")["effect_size_numeric"].mean().to_dict()
        final = _as_optional_float(means.get("final"))
        previous = _as_optional_float(means.get("previous_occurrence"))
        source = _as_optional_float(means.get("source_position"))
        distractor = _as_optional_float(means.get("distractor_position"))
        source_like = source if source is not None else previous
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_group": candidate_group,
                "layer": int(layer),
                "head": int(head),
                "final_effect": final,
                "previous_occurrence_effect": previous,
                "source_position_effect": source,
                "distractor_position_effect": distractor,
                "final_minus_source": (
                    final - source_like if final is not None and source_like is not None else None
                ),
                "source_minus_distractor": (
                    source_like - distractor
                    if source_like is not None and distractor is not None
                    else None
                ),
                "position_specificity_status": classify_position_specificity(
                    final_effect=final,
                    previous_occurrence_effect=previous,
                    source_position_effect=source,
                    distractor_position_effect=distractor,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "final_effect",
        ascending=False,
        na_position="last",
    )


def classify_position_specificity(
    *,
    final_effect: float | None,
    previous_occurrence_effect: float | None,
    source_position_effect: float | None,
    distractor_position_effect: float | None,
) -> str:
    final = _positive(final_effect)
    source = _positive(source_position_effect) or _positive(previous_occurrence_effect)
    distractor = _positive(distractor_position_effect)
    source_value = _first_number(source_position_effect, previous_occurrence_effect)
    positive_values = [
        value for value in [final_effect, previous_occurrence_effect, source_position_effect] if value is not None
    ]
    if not positive_values:
        return "insufficient_positions"
    if not final and not source:
        return "no_position_effect"
    if distractor and distractor_position_effect is not None:
        strongest_expected = max([value for value in positive_values if value is not None], default=0.0)
        if distractor_position_effect >= strongest_expected:
            return "distractor_like"
    if final and source:
        if (
            final_effect is not None
            and source_value is not None
            and abs(final_effect - source_value) < 0.05
        ):
            return "position_nonspecific"
        return "both_source_and_destination"
    if final:
        return "destination_specific"
    if source:
        return "source_specific"
    return "insufficient_positions"


def position_characterization_summary(
    results: pd.DataFrame,
    by_candidate: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "n_result_rows": int(len(results)),
        "n_candidates": int(by_candidate["candidate_id"].nunique()) if not by_candidate.empty else 0,
        "position_status_counts": (
            by_candidate["position_specificity_status"].value_counts().to_dict()
            if not by_candidate.empty
            else {}
        ),
        "interpretation_note": (
            "Position characterization asks where interventions matter. It does not by itself "
            "establish an induction-head mechanism."
        ),
    }


def write_position_characterization_artifacts(
    output_root: Path,
    results: pd.DataFrame,
    by_candidate: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    results.to_csv(output_root / "position_characterization_results.csv", index=False)
    by_candidate.to_csv(output_root / "position_characterization_by_candidate.csv", index=False)
    (output_root / "position_characterization_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "position_characterization.md").write_text(
        _position_markdown(summary, by_candidate),
        encoding="utf-8",
    )
    figures = output_root / "figures"
    figures.mkdir(exist_ok=True)
    for head_label in ["L7H7", "L9H11"]:
        layer, head = _parse_head_label(head_label)
        fig = plot_position_grid(
            results[(results["layer"] == layer) & (results["head"] == head)],
            head_label,
        )
        fig.savefig(figures / f"position_grid_{head_label}.png", dpi=160, bbox_inches="tight")
        fig.savefig(figures / f"position_grid_{head_label}.svg", format="svg", bbox_inches="tight")
        plt.close(fig)


def plot_position_grid(results: pd.DataFrame, head_label: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    if results.empty:
        ax.text(0.5, 0.5, "No rows", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    rows = results.copy()
    rows["effect_size_numeric"] = pd.to_numeric(rows["effect_size"], errors="coerce")
    grid = rows.pivot_table(
        index="family",
        columns="position_label",
        values="effect_size_numeric",
        aggfunc="mean",
    )
    image = ax.imshow(grid.fillna(0).values, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(grid.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index)
    ax.set_title(f"Position characterization: {head_label}")
    fig.colorbar(image, ax=ax, label="Mean effect size")
    fig.tight_layout()
    return fig


def _selected_records(
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


def _position_row(row: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    record: PromptRecord = job["record"]
    return {
        "seed": int(row["seed"]),
        "candidate_id": job["candidate_id"],
        "candidate_group": job["candidate_group"],
        "layer": int(row["layer"]),
        "head": int(row["head"]),
        "family": row["family"],
        "heldout_family_type": record.heldout_family_type,
        "example_id": row["example_id"],
        "intervention": row["intervention"],
        "position_label": row["position_label"],
        "position_status": row["position_status"],
        "head_specific_patch": row["head_specific_patch"],
        "actual_patch_scope": row["actual_patch_scope"],
        "metric": row["metric"],
        "effect_size": row["effect_size"],
        "effect_size_status": row["effect_size_status"],
        "clean_prompt": row["clean_prompt"],
        "corrupt_prompt": row["corrupt_prompt"],
    }


def _position_markdown(summary: dict[str, Any], by_candidate: pd.DataFrame) -> str:
    lines = [
        "# Position Characterization",
        "",
        f"- Rows: `{summary['n_result_rows']}`",
        f"- Candidates: `{summary['n_candidates']}`",
        f"- Position statuses: `{summary['position_status_counts']}`",
        "",
        "Position effects are local diagnostics, not mechanism claims.",
        "",
        "| candidate | head | status | final | source | distractor |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    if by_candidate.empty:
        lines.append("| none |  |  |  |  |  |")
    else:
        for row in by_candidate.head(12).itertuples(index=False):
            source = (
                row.source_position_effect
                if row.source_position_effect is not None and not pd.isna(row.source_position_effect)
                else row.previous_occurrence_effect
            )
            lines.append(
                f"| {row.candidate_id} | L{int(row.layer)}H{int(row.head)} | "
                f"{row.position_specificity_status} | {_fmt(row.final_effect)} | "
                f"{_fmt(source)} | {_fmt(row.distractor_position_effect)} |"
            )
    return "\n".join(lines) + "\n"


def _positive(value: float | None) -> bool:
    return value is not None and not pd.isna(value) and value > 0


def _first_number(*values: float | None) -> float | None:
    for value in values:
        if value is not None and not pd.isna(value):
            return float(value)
    return None


def _as_optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _parse_head_label(label: str) -> tuple[int, int]:
    normalized = label.strip().upper()
    layer_text, head_text = normalized[1:].split("H", maxsplit=1)
    return int(layer_text), int(head_text)


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.4f}"
