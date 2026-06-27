from __future__ import annotations

import argparse

from local_mi_lab.publication_figures import generate_head_specific_publication_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-dir",
        default="reports/head_specific_induction_causality_v1",
        help="Directory containing consolidated head-specific report artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/head_specific_induction_causality_v1",
        help="Tracked output directory for SVG/PDF figures.",
    )
    args = parser.parse_args()
    manifest = generate_head_specific_publication_figures(args.report_dir, args.output_dir)
    print(manifest["output_dir"])


if __name__ == "__main__":
    main()
