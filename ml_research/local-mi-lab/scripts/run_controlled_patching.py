from __future__ import annotations

import argparse

from local_mi_lab.config import load_config
from local_mi_lab.controlled_patching import (
    DEFAULT_CONTROLLED_PATCHING_FAMILIES,
    build_patching_jobs,
    run_controlled_patching,
)
from local_mi_lab.models import load_hooked_transformer
from local_mi_lab.prompts import read_prompts_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--examples-per-family", type=int, default=8)
    parser.add_argument("--families")
    parser.add_argument("--component", choices=["attn_out", "resid_post"])
    parser.add_argument("--position", default="final")
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    families = (
        [family.strip() for family in args.families.split(",") if family.strip()]
        if args.families
        else DEFAULT_CONTROLLED_PATCHING_FAMILIES
    )
    if args.dry_run:
        records = read_prompts_csv(f"{args.run}/prompts.csv")
        import pandas as pd

        candidates = pd.read_csv(args.candidates).to_dict(orient="records")
        jobs = build_patching_jobs(
            records,
            candidates,
            families=families,
            examples_per_family=args.examples_per_family,
            max_candidates=args.max_candidates,
            seed=args.seed,
            component_override=args.component,
            position_label=args.position,
        )
        print({"dry_run": True, "n_jobs": len(jobs), "families": families})
        return

    config = load_config(args.config)
    model = load_hooked_transformer(config)
    model.eval()
    summary = run_controlled_patching(
        model,
        args.run,
        args.candidates,
        families=families,
        examples_per_family=args.examples_per_family,
        max_candidates=args.max_candidates,
        seed=args.seed,
        component_override=args.component,
        position_label=args.position,
    )
    print(summary)


if __name__ == "__main__":
    main()
