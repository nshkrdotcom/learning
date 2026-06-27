from __future__ import annotations

import argparse

from local_mi_lab.candidate_characterization_report import (
    compare_candidate_characterization_runs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument(
        "--output",
        default="reports/head_specific_candidate_characterization_v1",
        help="Output directory for consolidated candidate characterization artifacts.",
    )
    parser.add_argument(
        "--tracked-summary",
        default="docs/results/head_specific_candidate_characterization_v1.md",
        help="Tracked Markdown summary path for ignored report artifacts.",
    )
    args = parser.parse_args()
    paths = compare_candidate_characterization_runs(
        args.runs,
        args.output,
        tracked_summary_path=args.tracked_summary,
    )
    print(paths["markdown"])
    if "tracked_summary" in paths:
        print(paths["tracked_summary"])


if __name__ == "__main__":
    main()
