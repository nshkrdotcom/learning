from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from local_mi_lab.config import load_config, max_examples_for_initial_run, write_config
from local_mi_lab.metrics import (
    aggregate_baseline,
    behavior_is_worth_activation_analysis,
    expected_token_stats,
)
from local_mi_lab.models import load_hooked_transformer
from local_mi_lab.paths import make_run_dir
from local_mi_lab.prompts import generate_induction_prompts, write_prompts_csv
from local_mi_lab.tokens import token_id_for_single_token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = make_run_dir(config["outputs"]["run_root"], config["experiment"]["name"])
    write_config(config, run_dir / "config.yaml")
    records = generate_induction_prompts(
        n_examples=max_examples_for_initial_run(config),
        seed=int(config["experiment"].get("seed", 0)),
    )
    write_prompts_csv(records, run_dir / "prompts.csv")
    rows, summary = run_baseline(config, records)
    pd.DataFrame(rows).to_csv(run_dir / "baseline_by_example.csv", index=False)
    (run_dir / "baseline_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_notes(run_dir, summary, rows)
    print(run_dir)


def run_baseline(config: dict[str, Any], records: list[Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = load_hooked_transformer(config)
    model.eval()
    rows: list[dict[str, Any]] = []
    for record in tqdm(records, desc="Baseline behavior"):
        target_id = token_id_for_single_token(model.tokenizer, record.expected_next_token)
        prompt_stats = _score_prompt(model, record.prompt, target_id)
        control_stats = _score_prompt(model, record.control_prompt, target_id)
        rows.append(
            {
                "example_id": record.example_id,
                "task": record.task,
                "prompt": record.prompt,
                "expected_next_token": record.expected_next_token,
                "expected_token_id": target_id,
                "expected_logit": prompt_stats["target_logit"],
                "expected_probability": prompt_stats["target_probability"],
                "expected_rank": prompt_stats["target_rank"],
                "control_prompt": record.control_prompt,
                "control_expected_logit": control_stats["target_logit"],
                "control_expected_probability": control_stats["target_probability"],
                "control_expected_rank": control_stats["target_rank"],
                "logit_diff_vs_control": prompt_stats["target_logit"]
                - control_stats["target_logit"],
                "probability_diff_vs_control": prompt_stats["target_probability"]
                - control_stats["target_probability"],
            }
        )
    summary = aggregate_baseline(rows)
    summary["model"] = config["model"]["name"]
    summary["task"] = config["task"]["name"]
    summary["behavior_analysis_note"] = (
        "This is behavior characterization, not interpretability evidence."
    )
    summary["activation_analysis_worth_running"] = behavior_is_worth_activation_analysis(summary)
    return rows, summary


def _score_prompt(model: Any, prompt: str, target_token_id: int) -> dict[str, float | int]:
    tokens = model.to_tokens(prompt)
    with torch.inference_mode():
        logits = model(tokens)
    return expected_token_stats(logits[0, -1, :], target_token_id)


def write_notes(run_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    failing = summary.get("failing_examples", [])
    lines = [
        "# Baseline Behavior Notes",
        "",
        "This is behavior characterization, not interpretability evidence.",
        "",
        "## Does the model show the behavior?",
        "",
        "Yes, if the expected-token probability is higher than the control and ranks are low for enough examples.",
        f"Activation analysis worth running: {summary['activation_analysis_worth_running']}.",
        "",
        "## How strong is it?",
        "",
        f"Mean expected probability: {summary['mean_expected_probability']}",
        f"Median expected rank: {summary['median_expected_rank']}",
        f"Mean probability diff versus control: {summary['mean_probability_diff_vs_control']}",
        "",
        "## Which examples fail?",
        "",
    ]
    if failing:
        for example_id in failing[:20]:
            row = next(row for row in rows if row["example_id"] == example_id)
            lines.append(
                f"- `{example_id}` rank={row['expected_rank']} prob_diff={row['probability_diff_vs_control']}"
            )
    else:
        lines.append("No failing examples under the current simple threshold.")
    lines.extend(
        [
            "",
            "## Is activation analysis worth running?",
            "",
            str(summary["activation_analysis_worth_running"]),
        ]
    )
    (run_dir / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
