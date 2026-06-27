from __future__ import annotations

import argparse

from local_mi_lab.config import load_config
from local_mi_lab.metric_calibration_report import run_metric_calibration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gpt2_small_induction_controls.yaml")
    parser.add_argument("--output", default="reports/induction_metric_calibration_v1")
    parser.add_argument(
        "--tracked-summary",
        default="docs/results/induction_metric_calibration_v1.md",
    )
    parser.add_argument(
        "--learning-note",
        default="docs/learning_notes/2026-06-26_induction_metric_calibration.md",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    paths = run_metric_calibration(
        config=config,
        output_dir=args.output,
        tracked_summary_path=args.tracked_summary,
        learning_note_path=args.learning_note,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
