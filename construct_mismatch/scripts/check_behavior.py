from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

from construct_mismatch.activations import get_logits, get_target_logit_diff
from construct_mismatch.datasets import (
    CONSTRUCTS,
    SPLITS,
    artifact_path,
    group_by_axis,
    load_dataset,
)
from construct_mismatch.metrics import (
    accuracy_from_signed,
    bootstrap_ci,
    signed_logit_diff,
    standard_error,
)
from construct_mismatch.model import load_model, prompt_to_tokens, token_id_for_target


def target_id_cache(model, records: list[dict[str, object]]) -> dict[str, int]:
    targets = {
        str(record["class_a_target"]) for record in records
    } | {str(record["class_b_target"]) for record in records}
    return {target: token_id_for_target(model, target) for target in targets}


def evaluate_records(model, records: list[dict[str, object]]) -> list[dict[str, object]]:
    token_ids = target_id_cache(model, records)
    rows: list[dict[str, object]] = []
    for record in tqdm(records, desc="behavior", leave=False):
        tokens = prompt_to_tokens(model, str(record["prompt"]))
        logits = get_logits(model, tokens)
        diff = float(
            get_target_logit_diff(
                logits,
                token_ids[str(record["class_a_target"])],
                token_ids[str(record["class_b_target"])],
            )[0]
        )
        signed = float(signed_logit_diff(np.asarray([diff]), [str(record["label"])])[0])
        if signed < -1.0:
            flag = "strong_model_disagreement"
        elif abs(signed) < 0.1:
            flag = "weak_margin_check_design"
        else:
            flag = ""
        rows.append(
            {
                "id": record["id"],
                "construct": record["construct"],
                "split": record["split"],
                "decoupling_axis": record["decoupling_axis"],
                "label": record["label"],
                "prompt": record["prompt"],
                "class_a_target": record["class_a_target"],
                "class_b_target": record["class_b_target"],
                "logit_diff": diff,
                "signed_logit_diff": signed,
                "correct": signed > 0.0,
                "flag": flag,
                "notes": record.get("notes", ""),
            }
        )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    by_group: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["construct"]), str(row["split"]), str(row["decoupling_axis"]))
        by_group.setdefault(key, []).append(row)
    for (construct, split, axis), group in sorted(by_group.items()):
        values = np.asarray([float(row["signed_logit_diff"]) for row in group], dtype=float)
        acc = accuracy_from_signed(values)
        ci_low, ci_high = bootstrap_ci(values, n_boot=1000, seed=0)
        usable = acc >= 0.55 and float(np.mean(values)) > 0.0
        summary.append(
            {
                "construct": construct,
                "split": split,
                "decoupling_axis": axis,
                "n": len(group),
                "accuracy": acc,
                "mean_signed_logit_diff": float(np.mean(values)),
                "signed_logit_diff_se": standard_error(values),
                "signed_logit_diff_ci_low": ci_low,
                "signed_logit_diff_ci_high": ci_high,
                "behavior_status": "usable" if usable else "behavior_absent_or_weak",
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--construct", choices=[*CONSTRUCTS, "all"], default="all")
    parser.add_argument("--output-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    model = load_model(args.model, args.device)
    constructs = CONSTRUCTS if args.construct == "all" else (args.construct,)

    all_example_rows: list[dict[str, object]] = []
    for construct in constructs:
        records: list[dict[str, object]] = []
        for split in SPLITS:
            split_records = load_dataset(construct, split, args.output_root)
            if split == "decoupling":
                for axis_records in group_by_axis(split_records).values():
                    records.extend(axis_records)
            else:
                records.extend(split_records)
        all_example_rows.extend(evaluate_records(model, records))

    summary_rows = summarize(all_example_rows)
    behavior_dir = artifact_path(args.output_root) / "behavior"
    write_csv(behavior_dir / "behavior_summary.csv", summary_rows)

    diagnostic_rows = [
        row
        for row in sorted(all_example_rows, key=lambda item: float(item["signed_logit_diff"]))
        if row["flag"]
    ]
    if not diagnostic_rows:
        diagnostic_rows = sorted(
            all_example_rows,
            key=lambda item: abs(float(item["signed_logit_diff"])),
        )[:20]
    write_csv(behavior_dir / "behavior_examples.csv", diagnostic_rows)


if __name__ == "__main__":
    main()
