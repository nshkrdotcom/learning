from __future__ import annotations

import argparse

from local_mi_lab.failure_taxonomy import build_failure_taxonomy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--counterexamples",
        default="reports/head_specific_candidate_characterization_v1/counterexamples",
    )
    parser.add_argument(
        "--summary",
        default="docs/results/head_specific_candidate_characterization_v1.md",
    )
    parser.add_argument(
        "--output",
        default="reports/head_specific_candidate_characterization_v1/failure_taxonomy",
    )
    parser.add_argument(
        "--tracked-summary",
        default="docs/results/head_specific_candidate_characterization_failure_taxonomy_v1.md",
    )
    args = parser.parse_args()
    paths = build_failure_taxonomy(
        counterexamples_dir=args.counterexamples,
        summary_path=args.summary,
        output_dir=args.output,
        tracked_summary_path=args.tracked_summary,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
