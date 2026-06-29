from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from construct_mismatch.activations import get_resid_activations
from construct_mismatch.datasets import CONSTRUCTS, artifact_path, group_by_axis, load_dataset
from construct_mismatch.directions import fit_diff_in_means, signed_projection_scores
from construct_mismatch.metrics import accuracy_from_signed, standard_error
from construct_mismatch.model import load_model, prompt_to_tokens
from construct_mismatch.probes import fit_logistic_probe, probe_signed_scores


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"status": "not_run"}]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def behavior_is_usable(root: Path, construct: str) -> bool:
    path = artifact_path(root) / "behavior" / "behavior_summary.csv"
    if not path.exists():
        print("Behavior summary not found; proceeding without behavior gate.")
        return True
    df = pd.read_csv(path)
    rows = df[
        (df["construct"] == construct)
        & (df["split"] == "heldout")
        & (df["decoupling_axis"] == "ordinary")
    ]
    if rows.empty:
        return True
    return str(rows.iloc[0]["behavior_status"]) == "usable"


def collect_final_activations(model, records: list[dict[str, object]]) -> dict[int, np.ndarray]:
    by_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in range(model.cfg.n_layers)}
    for record in tqdm(records, desc="resid activations", leave=False):
        tokens = prompt_to_tokens(model, str(record["prompt"]))
        activations = get_resid_activations(model, tokens, position="final")
        for layer, tensor in activations.items():
            by_layer[layer].append(tensor[0].detach().cpu().numpy())
    return {layer: np.stack(values, axis=0) for layer, values in by_layer.items()}


def evaluate(
    construct: str,
    split: str,
    axis: str,
    records: list[dict[str, object]],
    activations: dict[int, np.ndarray],
    probes: dict[int, object],
    direction_set,
) -> list[dict[str, object]]:
    labels = [str(record["label"]) for record in records]
    rows: list[dict[str, object]] = []
    for layer, acts in activations.items():
        probe_signed = probe_signed_scores(probes[layer], acts, labels)
        center = (direction_set.mean_a[layer] + direction_set.mean_b[layer]) / 2.0
        direction_scores = (acts - center) @ direction_set.directions[layer]
        direction_signed = signed_projection_scores(direction_scores, labels)
        rows.append(
            {
                "construct": construct,
                "method": "probe",
                "split": split,
                "decoupling_axis": axis,
                "layer": layer,
                "n": len(records),
                "accuracy": accuracy_from_signed(probe_signed),
                "mean_signed_probe_score": float(np.mean(probe_signed)),
                "signed_probe_score_se": standard_error(probe_signed),
                "direction_accuracy_reference": accuracy_from_signed(direction_signed),
                "direction_mean_signed_projection_reference": float(np.mean(direction_signed)),
            }
        )
    return rows


def plot_per_layer(metrics_path: Path, output_dir: Path, construct: str) -> None:
    df = pd.read_csv(metrics_path)
    if "accuracy" not in df:
        return
    heldout = df[(df["split"] == "heldout") & (df["decoupling_axis"] == "ordinary")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(heldout["layer"], heldout["accuracy"], marker="o", label="probe")
    ax.plot(
        heldout["layer"],
        heldout["direction_accuracy_reference"],
        marker="o",
        label="direction reference",
    )
    ax.set_title(f"{construct}: probe heldout accuracy by layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    fig.tight_layout()
    path = output_dir / f"{construct}_probe_per_layer.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_decoupling(metrics_path: Path, output_dir: Path, construct: str) -> None:
    df = pd.read_csv(metrics_path)
    if "accuracy" not in df:
        return
    heldout = df[(df["split"] == "heldout") & (df["decoupling_axis"] == "ordinary")]
    if heldout.empty:
        return
    selected_layer = int(heldout.sort_values("accuracy", ascending=False).iloc[0]["layer"])
    plot_df = df[df["layer"] == selected_layer].copy()
    plot_df = plot_df[plot_df["split"].isin(["heldout", "decoupling"])]
    labels = [
        "ordinary_heldout" if row.split == "heldout" else row.decoupling_axis
        for row in plot_df.itertuples()
    ]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, plot_df["accuracy"], width, label="probe")
    ax.bar(x + width / 2, plot_df["direction_accuracy_reference"], width, label="direction")
    ax.set_title(f"{construct}: probe decoupling at layer {selected_layer}")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.legend(loc="best")
    fig.tight_layout()
    path = output_dir / f"{construct}_probe_decoupling_by_axis.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Wrote {path}")


def placeholder_plot(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, title, ha="center", va="center")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--construct", choices=CONSTRUCTS, required=True)
    parser.add_argument("--output-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--ignore-behavior-gate", action="store_true")
    args = parser.parse_args()

    output_dir = artifact_path(args.output_root) / "probes"
    metrics_path = output_dir / f"{args.construct}_probe_metrics.csv"
    if not args.ignore_behavior_gate and not behavior_is_usable(args.output_root, args.construct):
        write_csv(metrics_path, [{"construct": args.construct, "method": "probe", "status": "behavior_absent"}])
        for name in ("probe_per_layer", "probe_decoupling_by_axis"):
            placeholder_plot(output_dir / f"{args.construct}_{name}.png", "behavior_absent")
        return

    model = load_model(args.model, args.device)
    train = load_dataset(args.construct, "train", args.output_root)
    heldout = load_dataset(args.construct, "heldout", args.output_root)
    decoupling = load_dataset(args.construct, "decoupling", args.output_root)

    train_acts = collect_final_activations(model, train)
    train_labels = [str(record["label"]) for record in train]
    probes = {
        layer: fit_logistic_probe(acts, train_labels, max_iter=args.max_iter)
        for layer, acts in train_acts.items()
    }
    direction_set = fit_diff_in_means(train_acts, train_labels)

    rows: list[dict[str, object]] = []
    eval_groups = [("train", "ordinary", train), ("heldout", "ordinary", heldout)]
    eval_groups.extend(("decoupling", axis, records) for axis, records in group_by_axis(decoupling).items())
    for split, axis, records in eval_groups:
        acts = train_acts if split == "train" else collect_final_activations(model, records)
        rows.extend(evaluate(args.construct, split, axis, records, acts, probes, direction_set))

    write_csv(metrics_path, rows)
    plot_per_layer(metrics_path, output_dir, args.construct)
    plot_decoupling(metrics_path, output_dir, args.construct)


if __name__ == "__main__":
    main()
