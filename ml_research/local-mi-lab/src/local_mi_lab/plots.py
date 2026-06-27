from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def plot_logit_lens(summary_df: Any, path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(
        summary_df["layer"],
        summary_df["mean_expected_probability"],
        marker="o",
        color="#0b6e4f",
        label="Mean probability",
    )
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Mean expected-token probability")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(
        summary_df["layer"],
        summary_df["median_expected_rank"],
        marker="s",
        color="#8f2d56",
        label="Median rank",
    )
    ax2.set_ylabel("Median expected-token rank")
    fig.suptitle("Logit lens expected-token summary")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_patching_heatmap(heatmap_df: Any, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    image = ax.imshow(heatmap_df.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xlabel("Patched position")
    ax.set_ylabel("Patched layer")
    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels([str(col) for col in heatmap_df.columns])
    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels([str(idx) for idx in heatmap_df.index])
    fig.colorbar(image, ax=ax, label="Effect size")
    fig.suptitle("Activation patching effect size")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_attention_induction_scores(df: Any, path: Path, top_k: int = 20) -> None:
    if df.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_title("Induction-like attention pattern candidates")
        ax.text(0.5, 0.5, "No attention rows", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return
    grouped = (
        df.groupby(["layer", "head"], as_index=False)
        .agg(mean_attention=("attention_to_previous_occurrence", "mean"))
        .sort_values("mean_attention", ascending=False)
        .head(top_k)
    )
    labels = [f"L{int(row.layer)}H{int(row.head)}" for row in grouped.itertuples()]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, grouped["mean_attention"], color="#0b6e4f")
    ax.set_xlabel("Head")
    ax.set_ylabel("Mean attention to previous occurrence")
    ax.set_title("Induction-like attention pattern candidates")
    ax.set_ylim(0, max(1.0, float(grouped["mean_attention"].max()) * 1.1) if not grouped.empty else 1.0)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_attention_by_family(by_family_df: Any, path: Path, top_k: int = 8) -> None:
    if by_family_df.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_title("Attention by family")
        ax.text(0.5, 0.5, "No family attention rows", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return

    positive = by_family_df[by_family_df["family"] == "positive_repeat_sequence"]
    top_heads = (
        positive.sort_values("mean_attention_to_previous_occurrence", ascending=False)
        .head(top_k)[["layer", "head"]]
        .drop_duplicates()
    )
    selected = by_family_df.merge(top_heads, on=["layer", "head"], how="inner")
    selected = selected.dropna(subset=["mean_attention_to_previous_occurrence"])
    labels = [f"L{int(row.layer)}H{int(row.head)}" for row in top_heads.itertuples()]
    families = list(selected["family"].drop_duplicates())

    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.8 / max(len(families), 1)
    x_positions = list(range(len(labels)))
    for family_index, family in enumerate(families):
        family_rows = selected[selected["family"] == family]
        values = []
        for row in top_heads.itertuples():
            match = family_rows[
                (family_rows["layer"] == row.layer) & (family_rows["head"] == row.head)
            ]
            values.append(
                float(match["mean_attention_to_previous_occurrence"].iloc[0])
                if not match.empty
                else 0.0
            )
        offsets = [x + (family_index - (len(families) - 1) / 2) * width for x in x_positions]
        ax.bar(offsets, values, width=width, label=family)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("Mean attention to previous occurrence")
    ax.set_title("Previous-occurrence attention by prompt family")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
