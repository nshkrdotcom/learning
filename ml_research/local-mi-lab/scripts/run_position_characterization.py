from __future__ import annotations

import argparse

from local_mi_lab.config import load_config
from local_mi_lab.models import load_hooked_transformer
from local_mi_lab.position_characterization import run_position_characterization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate-set", required=True)
    parser.add_argument("--families", default="")
    parser.add_argument("--examples-per-family", type=int, default=12)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    model = load_hooked_transformer(config)
    model.eval()
    families = [family for family in args.families.split(",") if family] or None
    summary = run_position_characterization(
        model,
        config,
        args.candidate_set,
        output_dir=args.output,
        families=families,
        examples_per_family=args.examples_per_family,
    )
    print(summary["n_result_rows"])


if __name__ == "__main__":
    main()
