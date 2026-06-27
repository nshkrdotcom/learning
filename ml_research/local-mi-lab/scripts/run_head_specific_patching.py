from __future__ import annotations

import argparse

from local_mi_lab.config import load_config
from local_mi_lab.head_patching import (
    DEFAULT_HEAD_PATCHING_FAMILIES,
    heads_from_candidate_csv,
    parse_heads,
    run_head_specific_patching,
)
from local_mi_lab.models import load_hooked_transformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--heads")
    parser.add_argument("--candidates")
    parser.add_argument("--candidate-source")
    parser.add_argument("--families", default=",".join(DEFAULT_HEAD_PATCHING_FAMILIES))
    parser.add_argument("--examples-per-family", type=int, default=8)
    parser.add_argument("--metric", default="true_vs_control_logit_diff")
    parser.add_argument(
        "--intervention",
        choices=["head_clean_to_corrupt_patch", "head_zero_ablation"],
        default="head_clean_to_corrupt_patch",
    )
    parser.add_argument("--position", default="final")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-heads", type=int)
    args = parser.parse_args()

    if not args.heads and not args.candidates:
        raise SystemExit("Pass either --heads or --candidates")
    if args.heads:
        heads = parse_heads(args.heads)
    else:
        heads = heads_from_candidate_csv(
            args.candidates,
            candidate_source=args.candidate_source,
            max_heads=args.max_heads,
        )
    if args.max_heads is not None:
        heads = heads[: args.max_heads]

    config = load_config(args.config)
    model = load_hooked_transformer(config)
    model.eval()
    summary = run_head_specific_patching(
        model,
        args.run,
        heads=heads,
        seed=args.seed,
        families=[family for family in args.families.split(",") if family],
        examples_per_family=args.examples_per_family,
        metric=args.metric,
        intervention=args.intervention,
        position_label=args.position,
    )
    print(summary)


if __name__ == "__main__":
    main()
