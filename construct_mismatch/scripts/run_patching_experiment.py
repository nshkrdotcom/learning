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

from construct_mismatch.datasets import (
    CONSTRUCTS,
    DECOUPLING_AXES,
    artifact_path,
    load_dataset,
    paired_records,
)
from construct_mismatch.model import load_model, prompt_to_tokens, token_id_for_target
from construct_mismatch.patching import patch_resid_pair, top_site_stability


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


def collect_pairs(
    construct: str,
    root: Path,
    max_pairs_per_axis: int,
) -> list[tuple[str, dict[str, object], dict[str, object]]]:
    heldout = load_dataset(construct, "heldout", root)
    decoupling = load_dataset(construct, "decoupling", root)
    selected: list[tuple[str, dict[str, object], dict[str, object]]] = []
    for clean, corrupt in paired_records(heldout, axis="ordinary")[:max_pairs_per_axis]:
        selected.append(("ordinary", clean, corrupt))
    for axis in DECOUPLING_AXES:
        for clean, corrupt in paired_records(decoupling, axis=axis)[:max_pairs_per_axis]:
            selected.append((axis, clean, corrupt))
    return selected


def plot_heatmap(metrics_path: Path, output_dir: Path, construct: str) -> None:
    df = pd.read_csv(metrics_path)
    path = output_dir / f"{construct}_patching_heatmap.png"
    if "recovery" not in df:
        placeholder_plot(path, "not_run")
        return
    grouped = df.groupby(["layer", "position"], as_index=False)["recovery"].mean()
    n_layers = int(grouped["layer"].max()) + 1
    n_positions = int(grouped["position"].max()) + 1
    heatmap = np.full((n_layers, n_positions), np.nan)
    for row in grouped.itertuples():
        heatmap[int(row.layer), int(row.position)] = float(row.recovery)
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(heatmap, aspect="auto", interpolation="nearest", cmap="coolwarm")
    ax.set_title(f"{construct}: residual patching recovery")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Layer")
    fig.colorbar(image, ax=ax, label="Mean recovery")
    fig.tight_layout()
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
    parser.add_argument("--max-pairs-per-axis", type=int, default=2)
    parser.add_argument("--ignore-behavior-gate", action="store_true")
    args = parser.parse_args()

    output_dir = artifact_path(args.output_root) / "patching"
    metrics_path = output_dir / f"{args.construct}_patching_metrics.csv"
    top_sites_path = output_dir / f"{args.construct}_top_sites.csv"

    if not args.ignore_behavior_gate and not behavior_is_usable(args.output_root, args.construct):
        rows = [{"construct": args.construct, "method": "patching", "status": "behavior_absent"}]
        write_csv(metrics_path, rows)
        write_csv(top_sites_path, rows)
        placeholder_plot(output_dir / f"{args.construct}_patching_heatmap.png", "behavior_absent")
        return

    model = load_model(args.model, args.device)
    pairs = collect_pairs(args.construct, args.output_root, args.max_pairs_per_axis)
    if not pairs:
        rows = [{"construct": args.construct, "method": "patching", "status": "not_run", "reason": "no_pairs"}]
        write_csv(metrics_path, rows)
        write_csv(top_sites_path, rows)
        placeholder_plot(output_dir / f"{args.construct}_patching_heatmap.png", "no_pairs")
        return

    target_ids = {
        target: token_id_for_target(model, target)
        for _, clean, corrupt in pairs
        for target in (
            str(clean["class_a_target"]),
            str(clean["class_b_target"]),
            str(corrupt["class_a_target"]),
            str(corrupt["class_b_target"]),
        )
    }

    metric_rows: list[dict[str, object]] = []
    for axis, clean, corrupt in tqdm(pairs, desc="patching pairs"):
        results = patch_resid_pair(
            model,
            prompt_to_tokens(model, str(clean["prompt"])),
            prompt_to_tokens(model, str(corrupt["prompt"])),
            target_ids[str(clean["class_a_target"])],
            target_ids[str(clean["class_b_target"])],
            pair_id=str(clean["pair_id"]),
            axis=axis,
        )
        for result in results:
            metric_rows.append(
                {
                    "construct": args.construct,
                    "method": "patching",
                    "pair_id": result.pair_id,
                    "decoupling_axis": result.axis,
                    "layer": result.layer,
                    "position": result.position,
                    "clean_diff": result.clean_diff,
                    "corrupt_diff": result.corrupt_diff,
                    "patched_diff": result.patched_diff,
                    "recovery": result.recovery,
                }
            )

    top_rows: list[dict[str, object]] = []
    by_pair: dict[str, list[dict[str, object]]] = {}
    for row in metric_rows:
        by_pair.setdefault(str(row["pair_id"]), []).append(row)
    for pair_id, rows in sorted(by_pair.items()):
        best = max(rows, key=lambda row: abs(float(row["recovery"])))
        axis_rows = [row for row in metric_rows if row["decoupling_axis"] == best["decoupling_axis"]]
        top_rows.append(
            {
                "construct": args.construct,
                "pair_id": pair_id,
                "decoupling_axis": best["decoupling_axis"],
                "top_layer": best["layer"],
                "top_position": best["position"],
                "top_recovery": best["recovery"],
                "axis_top_site_stability": top_site_stability(axis_rows),
            }
        )

    write_csv(metrics_path, metric_rows)
    write_csv(top_sites_path, top_rows)
    plot_heatmap(metrics_path, output_dir, args.construct)


if __name__ == "__main__":
    main()
