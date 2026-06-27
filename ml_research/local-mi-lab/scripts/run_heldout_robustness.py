from __future__ import annotations

import argparse

from local_mi_lab.config import load_config
from local_mi_lab.heldout_robustness import run_heldout_robustness
from local_mi_lab.models import load_hooked_transformer


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate-set", required=True)
    parser.add_argument("--families")
    parser.add_argument("--interventions")
    parser.add_argument("--positions")
    parser.add_argument("--examples-per-family", type=int, default=12)
    parser.add_argument("--metric", default="true_vs_control_logit_diff")
    args = parser.parse_args()

    config = load_config(args.config)
    model = load_hooked_transformer(config)
    model.eval()
    run_dir = run_heldout_robustness(
        model,
        config,
        args.candidate_set,
        families=_split_csv(args.families),
        interventions=_split_csv(args.interventions),
        positions=_split_csv(args.positions),
        examples_per_family=args.examples_per_family,
        metric=args.metric,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
