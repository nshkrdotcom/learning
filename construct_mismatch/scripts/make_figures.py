from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from construct_mismatch.datasets import artifact_path
from construct_mismatch.reporting import generate_report
from construct_mismatch.scoring import generate_scoring_artifacts


def tokenization_plot(root: Path) -> None:
    path = artifact_path(root) / "tokenization" / "gpt2_small_target_tokens.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    grouped = df.groupby("construct")["usable_as_target"].agg(["sum", "count"])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(grouped))
    ax.bar(x, grouped["sum"], label="usable")
    ax.bar(x, grouped["count"] - grouped["sum"], bottom=grouped["sum"], label="not usable")
    ax.set_xticks(x, grouped.index)
    ax.set_ylabel("Candidate targets")
    ax.set_title("GPT-2 target-token validation")
    ax.legend(loc="best")
    fig.tight_layout()
    out = artifact_path(root) / "tokenization" / "gpt2_small_target_tokens.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


def behavior_plot(root: Path) -> None:
    path = artifact_path(root) / "behavior" / "behavior_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df.copy()
    df["label"] = df["construct"] + "/" + df["split"] + "/" + df["decoupling_axis"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(df["label"], df["accuracy"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Behavior accuracy by construct and split")
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    fig.tight_layout()
    out = artifact_path(root) / "behavior" / "behavior_accuracy_by_construct_split.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    tokenization_plot(args.output_root)
    behavior_plot(args.output_root)
    matrix, classifications = generate_scoring_artifacts(args.output_root)
    del matrix, classifications
    report_path = generate_report(args.output_root)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
