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
