from __future__ import annotations

import argparse

from local_mi_lab.attention_effect_alignment import run_attention_effect_alignment
from local_mi_lab.config import load_config
from local_mi_lab.models import load_hooked_transformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--heldout-run", required=True)
    parser.add_argument("--candidate-set", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--example-limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model = load_hooked_transformer(config)
    model.eval()
    summary = run_attention_effect_alignment(
        model,
        heldout_run=args.heldout_run,
        candidate_set=args.candidate_set,
        output_dir=args.output,
        example_limit=args.example_limit,
    )
    print(summary["n_example_rows"])


if __name__ == "__main__":
    main()
