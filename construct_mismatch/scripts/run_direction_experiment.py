from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens.utilities import get_act_name

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from construct_mismatch.activations import get_resid_activations, get_target_logit_diff
from construct_mismatch.datasets import (
    CONSTRUCTS,
    artifact_path,
    group_by_axis,
    load_dataset,
)
from construct_mismatch.directions import (
    DirectionSet,
    fit_diff_in_means,
    random_direction_like,
    shuffled_label_direction,
    signed_projection_scores,
)
from construct_mismatch.metrics import (
    accuracy_from_signed,
    kl_divergence_from_logits,
    standard_error,
)
from construct_mismatch.model import load_model, prompt_to_tokens, token_id_for_target


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


def evaluate_direction_set(
    construct: str,
    split: str,
    axis: str,
    records: list[dict[str, object]],
    activations: dict[int, np.ndarray],
    direction_set: DirectionSet,
    baseline_type: str,
) -> list[dict[str, object]]:
    labels = [str(record["label"]) for record in records]
    rows: list[dict[str, object]] = []
    for layer, acts in activations.items():
        center = (direction_set.mean_a[layer] + direction_set.mean_b[layer]) / 2.0
        scores = (acts - center) @ direction_set.directions[layer]
        signed = signed_projection_scores(scores, labels)
        rows.append(
            {
                "construct": construct,
                "method": "direction",
                "baseline_type": baseline_type,
                "split": split,
                "decoupling_axis": axis,
                "layer": layer,
                "n": len(records),
                "accuracy": accuracy_from_signed(signed),
                "mean_signed_projection": float(np.mean(signed)),
                "signed_projection_se": standard_error(signed),
            }
        )
    return rows


def build_random_direction_set(direction_set: DirectionSet, seed: int = 0) -> DirectionSet:
    return DirectionSet(
        directions={
            layer: random_direction_like(direction, seed=seed + layer)
            for layer, direction in direction_set.directions.items()
        },
        mean_a=direction_set.mean_a,
        mean_b=direction_set.mean_b,
    )


@torch.no_grad()
def collect_steering_metrics(
    model,
    construct: str,
    records: list[dict[str, object]],
    direction_sets: dict[str, DirectionSet],
    alphas: list[float],
) -> list[dict[str, object]]:
    aggregates: dict[tuple[str, int, float], dict[str, list[float]]] = {}
    target_ids = {
        target: token_id_for_target(model, target)
        for record in records
        for target in (str(record["class_a_target"]), str(record["class_b_target"]))
    }
    for record in tqdm(records, desc="steering", leave=False):
        tokens = prompt_to_tokens(model, str(record["prompt"]))
        baseline_logits = model(tokens)
        class_a_id = target_ids[str(record["class_a_target"])]
        class_b_id = target_ids[str(record["class_b_target"])]
        baseline_diff = float(get_target_logit_diff(baseline_logits, class_a_id, class_b_id)[0])
        label_sign = 1.0 if record["label"] == "class_a" else -1.0
        baseline_signed = label_sign * baseline_diff
        for baseline_type, direction_set in direction_sets.items():
            for layer, direction in direction_set.directions.items():
                hook_name = get_act_name("resid_post", layer)
                direction_tensor = torch.tensor(direction, dtype=torch.float32)
                for alpha in alphas:

                    def steering_hook(
                        resid: torch.Tensor,
                        hook,
                        alpha: float = alpha,
                        direction_tensor: torch.Tensor = direction_tensor,
                    ) -> torch.Tensor:
                        resid[:, -1, :] = resid[:, -1, :] + alpha * direction_tensor.to(resid.device)
                        return resid

                    steered_logits = model.run_with_hooks(
                        tokens,
                        fwd_hooks=[(hook_name, steering_hook)],
                    )
                    steered_diff = float(
                        get_target_logit_diff(steered_logits, class_a_id, class_b_id)[0]
                    )
                    steered_signed = label_sign * steered_diff
                    key = (baseline_type, layer, alpha)
                    bucket = aggregates.setdefault(
                        key,
                        {
                            "raw_logit_shift": [],
                            "signed_logit_shift": [],
                            "steered_signed_logit_diff": [],
                            "kl_divergence": [],
                        },
                    )
                    bucket["raw_logit_shift"].append(steered_diff - baseline_diff)
                    bucket["signed_logit_shift"].append(steered_signed - baseline_signed)
                    bucket["steered_signed_logit_diff"].append(steered_signed)
                    bucket["kl_divergence"].append(
                        kl_divergence_from_logits(
                            baseline_logits[:, -1, :],
                            steered_logits[:, -1, :],
                        )
                    )
    rows: list[dict[str, object]] = []
    for (baseline_type, layer, alpha), values in sorted(aggregates.items()):
        rows.append(
            {
                "construct": construct,
                "method": "direction",
                "baseline_type": baseline_type,
                "split": "heldout",
                "decoupling_axis": "ordinary",
                "layer": layer,
                "alpha": alpha,
                "n": len(values["raw_logit_shift"]),
                "mean_raw_logit_shift": float(np.mean(values["raw_logit_shift"])),
                "mean_abs_raw_logit_shift": float(np.mean(np.abs(values["raw_logit_shift"]))),
                "mean_signed_logit_shift": float(np.mean(values["signed_logit_shift"])),
                "mean_steered_signed_logit_diff": float(np.mean(values["steered_signed_logit_diff"])),
                "mean_kl_divergence": float(np.mean(values["kl_divergence"])),
            }
        )
    return rows


def plot_prediction(metrics_path: Path, output_dir: Path, construct: str) -> None:
    df = pd.read_csv(metrics_path)
    if "accuracy" not in df:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for axis, group in df[
        (df["baseline_type"] == "direction")
        & (
            ((df["split"] == "heldout") & (df["decoupling_axis"] == "ordinary"))
            | (df["split"] == "decoupling")
        )
    ].groupby("decoupling_axis"):
        ax.plot(group["layer"], group["accuracy"], marker="o", label=axis)
    ax.set_title(f"{construct}: direction prediction by layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = output_dir / f"{construct}_per_layer_prediction.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_steering(steering_path: Path, output_dir: Path, construct: str) -> None:
    df = pd.read_csv(steering_path)
    if "mean_abs_raw_logit_shift" not in df:
        return
    df = df[(df["baseline_type"] == "direction") & (df["alpha"].abs() == df["alpha"].abs().max())]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    for alpha, group in df.groupby("alpha"):
        ax1.plot(group["layer"], group["mean_abs_raw_logit_shift"], marker="o", label=f"abs shift alpha={alpha:g}")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Mean abs target-logit shift")
    ax2 = ax1.twinx()
    kl_group = df.groupby("layer", as_index=False)["mean_kl_divergence"].mean()
    ax2.plot(kl_group["layer"], kl_group["mean_kl_divergence"], color="black", linestyle="--", label="KL")
    ax2.set_ylabel("Mean KL divergence")
    ax1.set_title(f"{construct}: direction steering by layer")
    ax1.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = output_dir / f"{construct}_per_layer_steering.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_decoupling(metrics_path: Path, output_dir: Path, construct: str) -> None:
    df = pd.read_csv(metrics_path)
    if "accuracy" not in df:
        return
    heldout = df[
        (df["baseline_type"] == "direction")
        & (df["split"] == "heldout")
        & (df["decoupling_axis"] == "ordinary")
    ]
    if heldout.empty:
        return
    selected_layer = int(heldout.sort_values("accuracy", ascending=False).iloc[0]["layer"])
    plot_df = df[(df["baseline_type"] == "direction") & (df["layer"] == selected_layer)]
    plot_df = plot_df[plot_df["split"].isin(["heldout", "decoupling"])]
    labels = [
        "ordinary_heldout" if row.split == "heldout" else row.decoupling_axis
        for row in plot_df.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(labels, plot_df["accuracy"])
    ax.set_title(f"{construct}: decoupling accuracy at layer {selected_layer}")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    path = output_dir / f"{construct}_decoupling_by_axis.png"
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
    parser.add_argument("--alphas", nargs="*", type=float, default=[-3.0, -1.0, 0.0, 1.0, 3.0])
    parser.add_argument("--max-steering-examples", type=int, default=24)
    parser.add_argument("--ignore-behavior-gate", action="store_true")
    args = parser.parse_args()

    output_dir = artifact_path(args.output_root) / "directions"
    metrics_path = output_dir / f"{args.construct}_direction_metrics.csv"
    steering_path = output_dir / f"{args.construct}_steering_metrics.csv"

    if not args.ignore_behavior_gate and not behavior_is_usable(args.output_root, args.construct):
        rows = [{"construct": args.construct, "method": "direction", "status": "behavior_absent"}]
        write_csv(metrics_path, rows)
        write_csv(steering_path, rows)
        for name in ("per_layer_prediction", "per_layer_steering", "decoupling_by_axis"):
            placeholder_plot(output_dir / f"{args.construct}_{name}.png", "behavior_absent")
        return

    model = load_model(args.model, args.device)
    train = load_dataset(args.construct, "train", args.output_root)
    heldout = load_dataset(args.construct, "heldout", args.output_root)
    decoupling = load_dataset(args.construct, "decoupling", args.output_root)

    train_acts = collect_final_activations(model, train)
    train_labels = [str(record["label"]) for record in train]
    direction_set = fit_diff_in_means(train_acts, train_labels)
    direction_sets = {
        "direction": direction_set,
        "random_direction": build_random_direction_set(direction_set, seed=17),
        "shuffled_label_direction": shuffled_label_direction(train_acts, train_labels, seed=23),
    }

    metric_rows: list[dict[str, object]] = []
    eval_groups = [("train", "ordinary", train), ("heldout", "ordinary", heldout)]
    eval_groups.extend(("decoupling", axis, records) for axis, records in group_by_axis(decoupling).items())
    for split, axis, records in eval_groups:
        acts = train_acts if split == "train" else collect_final_activations(model, records)
        for baseline_type, candidate_set in direction_sets.items():
            metric_rows.extend(
                evaluate_direction_set(
                    args.construct,
                    split,
                    axis,
                    records,
                    acts,
                    candidate_set,
                    baseline_type,
                )
            )
    write_csv(metrics_path, metric_rows)

    steering_records = heldout[: args.max_steering_examples]
    steering_rows = collect_steering_metrics(
        model,
        args.construct,
        steering_records,
        direction_sets,
        args.alphas,
    )
    write_csv(steering_path, steering_rows)

    plot_prediction(metrics_path, output_dir, args.construct)
    plot_steering(steering_path, output_dir, args.construct)
    plot_decoupling(metrics_path, output_dir, args.construct)


if __name__ == "__main__":
    main()
