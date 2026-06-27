from __future__ import annotations

import argparse

from local_mi_lab.config import load_config
from local_mi_lab.head_circuit_diagnostics import run_head_circuit_diagnostics
from local_mi_lab.models import load_hooked_transformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate-set", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--examples-per-family", type=int, default=12)
    args = parser.parse_args()

    config = load_config(args.config)
    model = load_hooked_transformer(config)
    model.eval()
    summary = run_head_circuit_diagnostics(
        model,
        config,
        args.candidate_set,
        output_dir=args.output,
        examples_per_family=args.examples_per_family,
    )
    print(summary["n_candidates"])


if __name__ == "__main__":
    main()
