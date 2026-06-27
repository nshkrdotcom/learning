from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from local_mi_lab.paths import resolve_repo_path

MULTISEED_CANDIDATE_COLUMNS = [
    "candidate_id",
    "candidate_group",
    "layer",
    "head",
    "n_seeds",
    "seeds_present",
    "n_survived_seeds",
    "n_downgraded_seeds",
    "n_falsified_seeds",
    "mean_positive_minus_control_gap",
    "min_positive_minus_control_gap",
    "families_survived_count",
    "interventions_survived",
    "positions_survived",
    "heldout_replication_status",
]

CONTROL_MOVING_STATUS = "falsified_controls_move"
SURVIVAL_STATUS = "heldout_survives_seed"
DOWNGRADE_STATUS = "downgraded_weak_family_specific"
NO_EFFECT_STATUS = "falsified_no_positive_effect"
SIGN_FLIP_STATUS = "falsified_sign_flip"
INSUFFICIENT_STATUS = "insufficient_valid_examples"


def load_heldout_run(run_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root = resolve_repo_path(run_dir)
    by_candidate_path = root / "heldout_robustness_by_candidate.csv"
    by_family_path = root / "heldout_robustness_by_family.csv"
    summary_path = root / "heldout_robustness_summary.json"
    if not by_candidate_path.exists():
        raise FileNotFoundError(f"Missing {by_candidate_path}")
    if not by_family_path.exists():
        raise FileNotFoundError(f"Missing {by_family_path}")
    by_candidate = pd.read_csv(by_candidate_path)
    by_family = pd.read_csv(by_family_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    seed = int(by_candidate["seed"].iloc[0]) if not by_candidate.empty else None
    manifest = {
        "run_dir": str(root),
        "seed": seed,
        "n_candidate_rows": int(len(by_candidate)),
        "n_family_rows": int(len(by_family)),
        "summary": summary,
    }
    return by_candidate, by_family, manifest


def compare_heldout_runs(run_dirs: list[str | Path], output_dir: str | Path) -> dict[str, Path]:
    root = resolve_repo_path(output_dir)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    candidate_frames: list[pd.DataFrame] = []
    family_frames: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        by_candidate, by_family, manifest = load_heldout_run(run_dir)
        candidate_frames.append(by_candidate)
        family_frames.append(by_family)
        manifests.append(manifest)

    candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    families = pd.concat(family_frames, ignore_index=True) if family_frames else pd.DataFrame()
    by_candidate = summarize_heldout_candidates(candidates, families)
    by_family = summarize_heldout_families(families)
    summary = heldout_multiseed_summary(by_candidate, by_family, manifests)

    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "run_manifest.json"
    by_candidate_path = root / "heldout_multiseed_by_candidate.csv"
    by_family_path = root / "heldout_multiseed_by_family.csv"
    summary_path = root / "heldout_multiseed_summary.json"
    markdown_path = root / "head_specific_induction_heldout_robustness_v1.md"
    candidate_fig = figures / "heldout_multiseed_candidate_gaps.png"
    status_fig = figures / "heldout_survival_status_counts.png"
    family_fig = figures / "heldout_family_failure_modes.png"

    manifest_path.write_text(json.dumps({"runs": manifests}, indent=2) + "\n", encoding="utf-8")
    by_candidate.to_csv(by_candidate_path, index=False)
    by_family.to_csv(by_family_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_heldout_markdown(summary, by_candidate, by_family), encoding="utf-8")
    plot_candidate_gaps(by_candidate, candidate_fig)
    plot_status_counts(by_candidate, status_fig)
    plot_family_failure_modes(by_family, family_fig)
    return {
        "manifest": manifest_path,
        "by_candidate": by_candidate_path,
        "by_family": by_family_path,
        "summary": summary_path,
        "markdown": markdown_path,
        "candidate_figure": candidate_fig,
        "status_figure": status_fig,
        "family_figure": family_fig,
    }


def summarize_heldout_candidates(candidates: pd.DataFrame, families: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=MULTISEED_CANDIDATE_COLUMNS)
    rows = candidates.copy()
    rows["positive_minus_control_gap"] = pd.to_numeric(
        rows["positive_minus_control_gap"],
        errors="coerce",
    )
    summaries: list[dict[str, Any]] = []
    for (candidate_id, layer, head), group in rows.groupby(["candidate_id", "layer", "head"], sort=True):
        seeds = sorted(int(seed) for seed in group["seed"].dropna().unique())
        survived = group[group["survival_status"] == SURVIVAL_STATUS]
        downgraded = group[group["survival_status"] == DOWNGRADE_STATUS]
        falsified = group[
            group["survival_status"].isin([CONTROL_MOVING_STATUS, NO_EFFECT_STATUS, SIGN_FLIP_STATUS])
        ]
        survived_families = _survived_positive_families(families, survived)
        status = classify_heldout_replication_status(group, survived_families)
        finite_gaps = group["positive_minus_control_gap"].dropna()
        summaries.append(
            {
                "candidate_id": str(candidate_id),
                "candidate_group": str(group["candidate_group"].iloc[0]),
                "layer": int(layer),
                "head": int(head),
                "n_seeds": len(seeds),
                "seeds_present": ",".join(str(seed) for seed in seeds),
                "n_survived_seeds": int(survived["seed"].nunique()),
                "n_downgraded_seeds": int(downgraded["seed"].nunique()),
                "n_falsified_seeds": int(falsified["seed"].nunique()),
                "mean_positive_minus_control_gap": (
                    float(finite_gaps.mean()) if not finite_gaps.empty else None
                ),
                "min_positive_minus_control_gap": (
                    float(finite_gaps.min()) if not finite_gaps.empty else None
                ),
                "families_survived_count": len(survived_families),
                "interventions_survived": ",".join(sorted(survived["intervention"].unique())),
                "positions_survived": ",".join(sorted(survived["position_label"].unique())),
                "heldout_replication_status": status,
            }
        )
    return pd.DataFrame(summaries, columns=MULTISEED_CANDIDATE_COLUMNS).sort_values(
        ["heldout_replication_status", "mean_positive_minus_control_gap"],
        ascending=[True, False],
        na_position="last",
    )


def classify_heldout_replication_status(
    group: pd.DataFrame,
    survived_positive_families: set[str] | None = None,
) -> str:
    if not bool(group["head_specific_patch"].all()):
        return "not_head_specific"
    seeds = set(int(seed) for seed in group["seed"].dropna().unique())
    if len(seeds) < 2:
        return "insufficient_heldout_data"
    statuses = set(group["survival_status"].dropna())
    if CONTROL_MOVING_STATUS in statuses:
        return "heldout_falsified"
    survived = group[group["survival_status"] == SURVIVAL_STATUS]
    survived_seeds = set(int(seed) for seed in survived["seed"].dropna().unique())
    survived_interventions = set(str(value) for value in survived["intervention"].dropna().unique())
    survived_positions = set(str(value) for value in survived["position_label"].dropna().unique())
    families_count = len(survived_positive_families or set())
    clean_final_survived_seeds = set(
        int(seed)
        for seed in survived[
            (survived["intervention"] == "head_clean_to_corrupt_patch")
            & (survived["position_label"] == "final")
        ]["seed"].dropna().unique()
    )
    if (
        len(survived_seeds) >= 2
        and families_count >= 2
        and len(clean_final_survived_seeds) >= 2
    ):
        if len(survived_interventions) < 2 or len(survived_positions) < 2:
            return "heldout_downgraded"
        return "heldout_replicated"
    if len(survived_seeds) >= 1 or DOWNGRADE_STATUS in statuses:
        return "heldout_downgraded"
    if statuses and statuses.issubset({INSUFFICIENT_STATUS}):
        return "insufficient_heldout_data"
    if NO_EFFECT_STATUS in statuses or SIGN_FLIP_STATUS in statuses:
        return "heldout_falsified"
    return "heldout_downgraded"


def summarize_heldout_families(families: pd.DataFrame) -> pd.DataFrame:
    if families.empty:
        return pd.DataFrame()
    rows = families.copy()
    rows["mean_effect_size"] = pd.to_numeric(rows["mean_effect_size"], errors="coerce")
    return (
        rows.groupby(["family", "heldout_family_type"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            n_candidate_rows=("candidate_id", "count"),
            mean_effect_size=("mean_effect_size", "mean"),
            max_effect_size=("mean_effect_size", "max"),
            n_position_unavailable=("n_position_unavailable", "sum"),
            n_denominator_zero=("n_denominator_zero", "sum"),
        )
        .sort_values(["heldout_family_type", "mean_effect_size"], ascending=[True, False])
    )


def heldout_multiseed_summary(
    by_candidate: pd.DataFrame,
    by_family: pd.DataFrame,
    manifests: list[dict[str, Any]],
) -> dict[str, Any]:
    status_counts = (
        by_candidate["heldout_replication_status"].value_counts().to_dict()
        if not by_candidate.empty
        else {}
    )
    replicated = by_candidate[
        by_candidate["heldout_replication_status"] == "heldout_replicated"
    ].copy()
    primary = by_candidate[
        by_candidate["candidate_group"].isin(["replicated_candidate", "random_comparison_replicated"])
    ].copy()
    raw = by_candidate[by_candidate["candidate_group"] == "prior_raw_attention_failed"].copy()
    negative = by_candidate[by_candidate["candidate_group"].str.startswith("negative_control")].copy()
    executive = (
        "At least one candidate survived the pre-registered held-out rule as a narrow local "
        "candidate. This is still not an induction-head discovery or a circuit claim."
        if not replicated.empty
        else "The held-out robustness check falsified or downgraded the previously replicated "
        "candidates. They should not be treated as induction-head candidates beyond the "
        "original synthetic setup."
    )
    return {
        "n_runs": len(manifests),
        "seeds": [manifest["seed"] for manifest in manifests],
        "n_candidates": int(len(by_candidate)),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "replicated_candidates": _candidate_rows(replicated),
        "primary_candidate_outcomes": _candidate_rows(primary),
        "prior_raw_attention_outcomes": _candidate_rows(raw),
        "negative_control_outcomes": _candidate_rows(negative),
        "family_summary": by_family.to_dict("records") if not by_family.empty else [],
        "executive_summary": executive,
        "limitation": (
            "This report compares fixed GPT-2 small heads under held-out synthetic prompt "
            "families, hook_z interventions, and the true-vs-control logit-diff metric. It is "
            "not broad model evidence."
        ),
    }


def render_heldout_markdown(
    summary: dict[str, Any],
    by_candidate: pd.DataFrame,
    by_family: pd.DataFrame,
) -> str:
    primary = _status_lines(summary["primary_candidate_outcomes"])
    raw = _status_lines(summary["prior_raw_attention_outcomes"])
    negative = _status_lines(summary["negative_control_outcomes"])
    replicated = _status_lines(summary["replicated_candidates"])
    family_lines = _family_lines(by_family)
    return "\n".join(
        [
            "# Head-Specific Induction Held-Out Robustness v1",
            "",
            "## Executive summary",
            "",
            summary["executive_summary"],
            "",
            "This is local MI practice evidence. It is not a mechanism claim, a circuit claim, "
            "or broad GPT-2 behavior.",
            "",
            "## Prior candidate result",
            "",
            "The prior head-specific sweep found narrow replicated candidates on one synthetic "
            "prompt generator. L7H7 was especially important to test because it was previously "
            "a random-comparison candidate.",
            "",
            "## Held-out design",
            "",
            "- Model: `gpt2-small`",
            "- Primary metric: `true_vs_control_logit_diff`",
            "- Hook: `blocks.<layer>.attn.hook_z`",
            "- Interventions: clean-to-corrupt patching, zero ablation, and mean ablation",
            "- Positions: final and previous occurrence",
            "- Seeds: " + ", ".join(str(seed) for seed in summary["seeds"]),
            "",
            "## Fixed candidate set",
            "",
            "The held-out runs used a fixed candidate set selected from prior artifacts before "
            "held-out scoring. No heads were selected from held-out outcomes.",
            "",
            "## Prompt families",
            "",
            "\n".join(family_lines) if family_lines else "No family rows were available.",
            "",
            "## Interventions and positions",
            "",
            "Final-position rows were available for all interventions. Previous-occurrence rows "
            "were frequently unavailable when the held-out control metadata had no meaningful "
            "source position; those rows are reported as insufficient rather than filled in.",
            "",
            "## Seed-level results",
            "",
            f"Status counts: `{summary['status_counts']}`",
            "",
            "## Candidate-level survival",
            "",
            "\n".join(primary) if primary else "No primary candidate outcomes were available.",
            "",
            "## L7H7",
            "",
            _candidate_paragraph(by_candidate, 7, 7),
            "",
            "## L9H11",
            "",
            _candidate_paragraph(by_candidate, 9, 11),
            "",
            "## Other replicated candidates",
            "",
            "\n".join(
                _status_lines(
                    [
                        row
                        for row in summary["primary_candidate_outcomes"]
                        if not (row["layer"] == 7 and row["head"] == 7)
                        and not (row["layer"] == 9 and row["head"] == 11)
                    ]
                )
            )
            or "No other primary candidate outcomes were available.",
            "",
            "## Prior raw-attention heads",
            "",
            "\n".join(raw) if raw else "No prior raw-attention comparison heads were available.",
            "",
            "## Negative controls",
            "",
            "\n".join(negative) if negative else "No negative-control outcomes were available.",
            "",
            "## Counterexamples",
            "",
            "Counterexample inspection should focus on candidates that were downgraded or "
            "falsified, especially when controls moved or intervention variants disagreed.",
            "",
            "## What survived",
            "",
            "\n".join(replicated) if replicated else "No candidate cleanly survived the held-out rule.",
            "",
            "## What failed",
            "",
            "Candidates with controls-moving failures, no-positive-effect rows, intervention-only "
            "effects, or unavailable previous-occurrence rows are downgraded or falsified for "
            "this lab stage.",
            "",
            "## What this teaches",
            "",
            "Held-out construction, causal controls, intervention variants, and position variants "
            "can all break a result that looked replicated under one generator.",
            "",
            "## What this does not show",
            "",
            "This does not discover an induction head, identify a circuit, or establish broad "
            "GPT-2 behavior. Even a surviving candidate remains local to the tested prompts, "
            "metric, hook, intervention, and positions.",
            "",
            "## Recommendation",
            "",
            _recommendation(summary),
            "",
        ]
    )


def _survived_positive_families(families: pd.DataFrame, survived: pd.DataFrame) -> set[str]:
    if families.empty or survived.empty:
        return set()
    keys = survived[["seed", "candidate_id", "intervention", "position_label"]].drop_duplicates()
    merged = families.merge(keys, on=["seed", "candidate_id", "intervention", "position_label"])
    positive = merged[
        (merged["heldout_family_type"] == "positive")
        & (pd.to_numeric(merged["mean_effect_size"], errors="coerce") > 0)
    ]
    return set(str(family) for family in positive["family"].dropna().unique())


def _candidate_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows = frame.copy()
    rows = rows.sort_values("mean_positive_minus_control_gap", ascending=False, na_position="last")
    out = []
    for row in rows.itertuples(index=False):
        out.append(
            {
                "candidate_id": row.candidate_id,
                "candidate_group": row.candidate_group,
                "layer": int(row.layer),
                "head": int(row.head),
                "mean_positive_minus_control_gap": _maybe_float(row.mean_positive_minus_control_gap),
                "min_positive_minus_control_gap": _maybe_float(row.min_positive_minus_control_gap),
                "n_survived_seeds": int(row.n_survived_seeds),
                "families_survived_count": int(row.families_survived_count),
                "interventions_survived": row.interventions_survived,
                "positions_survived": row.positions_survived,
                "heldout_replication_status": row.heldout_replication_status,
            }
        )
    return out


def _status_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = []
    for row in rows:
        lines.append(
            "- L{layer}H{head}: `{status}`, survived seeds `{seeds}`, mean gap `{gap}` "
            "({group}).".format(
                layer=row["layer"],
                head=row["head"],
                status=row["heldout_replication_status"],
                seeds=row["n_survived_seeds"],
                gap=_fmt(row["mean_positive_minus_control_gap"]),
                group=row["candidate_group"],
            )
        )
    return lines


def _candidate_paragraph(by_candidate: pd.DataFrame, layer: int, head: int) -> str:
    row = by_candidate[(by_candidate["layer"] == layer) & (by_candidate["head"] == head)]
    if row.empty:
        return f"L{layer}H{head} was not present in the held-out candidate table."
    first = row.iloc[0]
    return (
        f"L{layer}H{head} is classified as `{first['heldout_replication_status']}`. "
        f"It survived {int(first['n_survived_seeds'])} seed(s), with mean gap "
        f"`{_fmt(first['mean_positive_minus_control_gap'])}` and survived interventions "
        f"`{first['interventions_survived']}`. This is not a mechanism claim."
    )


def _family_lines(by_family: pd.DataFrame) -> list[str]:
    if by_family.empty:
        return []
    return [
        f"- {row.family}: mean effect `{_fmt(row.mean_effect_size)}`, type `{row.heldout_family_type}`."
        for row in by_family.itertuples(index=False)
    ]


def _recommendation(summary: dict[str, Any]) -> str:
    if summary["replicated_candidates"]:
        return (
            "Inspect surviving candidates manually and run a smaller, stricter replication before "
            "any stronger claim."
        )
    return (
        "Treat the prior replicated candidates as downgraded or falsified for this lab stage, "
        "then write the learning note before adding any new model or task."
    )


def plot_candidate_gaps(by_candidate: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if by_candidate.empty:
        ax.text(0.5, 0.5, "No candidate rows", ha="center", va="center")
        ax.axis("off")
    else:
        top = by_candidate.sort_values(
            "mean_positive_minus_control_gap",
            ascending=False,
            na_position="last",
        ).head(20)
        labels = [f"L{int(row.layer)}H{int(row.head)}" for row in top.itertuples()]
        values = pd.to_numeric(top["mean_positive_minus_control_gap"], errors="coerce").fillna(0)
        colors = [
            "#1f77b4" if status == "heldout_replicated" else "#8c8c8c"
            for status in top["heldout_replication_status"]
        ]
        ax.bar(labels, values, color=colors)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel("Mean positive-minus-control gap")
        ax.set_title("Held-out multi-seed candidate gaps")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_status_counts(by_candidate: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if by_candidate.empty:
        ax.text(0.5, 0.5, "No candidate rows", ha="center", va="center")
        ax.axis("off")
    else:
        counts = by_candidate["heldout_replication_status"].value_counts()
        ax.bar(counts.index, counts.values, color="#6a4c93")
        ax.set_ylabel("Candidates")
        ax.set_title("Held-out replication status counts")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_family_failure_modes(by_family: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if by_family.empty:
        ax.text(0.5, 0.5, "No family rows", ha="center", va="center")
        ax.axis("off")
    else:
        rows = by_family.copy()
        ax.bar(rows["family"], rows["mean_effect_size"], color="#0b6e4f")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel("Mean effect size")
        ax.set_title("Held-out family effects")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _maybe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.4f}"
