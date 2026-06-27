from __future__ import annotations

import argparse

from local_mi_lab.heldout_robustness_report import compare_heldout_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument(
        "--output",
        default="reports/head_specific_induction_heldout_robustness_v1",
        help="Output directory for the consolidated held-out robustness report.",
    )
    args = parser.parse_args()
    paths = compare_heldout_runs(args.runs, args.output)
    print(paths["markdown"])


if __name__ == "__main__":
    main()
