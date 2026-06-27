from __future__ import annotations

import argparse

from local_mi_lab.head_specific_report import compare_head_specific_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument(
        "--output",
        default="reports/head_specific_induction_causality_v1",
        help="Output directory for the consolidated report.",
    )
    args = parser.parse_args()
    paths = compare_head_specific_runs(args.runs, args.output)
    print(paths["markdown"])


if __name__ == "__main__":
    main()
