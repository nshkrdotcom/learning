from __future__ import annotations

import argparse

from local_mi_lab.candidate_characterization_counterexamples import (
    inspect_candidate_characterization_counterexamples,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--report-dir",
        default="reports/head_specific_candidate_characterization_v1",
    )
    parser.add_argument(
        "--output",
        default="reports/head_specific_candidate_characterization_v1/counterexamples",
    )
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    paths = inspect_candidate_characterization_counterexamples(
        candidate=args.candidate,
        report_dir=args.report_dir,
        output_dir=args.output,
        top_k=args.top_k,
    )
    print(paths["markdown"])


if __name__ == "__main__":
    main()
