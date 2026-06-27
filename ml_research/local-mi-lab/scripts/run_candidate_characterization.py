from __future__ import annotations

import argparse
from pathlib import Path

from local_mi_lab.candidate_characterization import run_candidate_characterization
from local_mi_lab.config import load_config
from local_mi_lab.models import load_hooked_transformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate-set", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--examples-per-family", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if args.overwrite and output.exists():
        for path in sorted(output.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    config = load_config(args.config)
    model = load_hooked_transformer(config)
    model.eval()
    summary = run_candidate_characterization(
        model,
        config,
        args.candidate_set,
        output_dir=args.output,
        examples_per_family=args.examples_per_family,
    )
    print(summary["characterization_status_counts"])


if __name__ == "__main__":
    main()
