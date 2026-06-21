from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from self_ground.real_residual_intervention import run_real_residual_intervention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a real TransformerLens residual-dimension smoke patch diagnostic. "
            "Outputs are diagnostic-only and claim-ineligible."
        )
    )
    parser.add_argument("--ranking-dir", default=None)
    parser.add_argument("--pairs", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-family", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--hook-point", default="blocks.2.hook_resid_post")
    parser.add_argument("--top-k-features", type=int, default=5)
    parser.add_argument("--operation", choices=["zero", "amplify"], default="zero")
    parser.add_argument("--factor", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.operation == "amplify" and args.factor == 1.0:
        print("ERROR: --operation amplify requires --factor not equal to 1.0", file=sys.stderr)
        return 1
    try:
        result = run_real_residual_intervention(
            out_dir=Path(args.out),
            ranking_dir=Path(args.ranking_dir) if args.ranking_dir else None,
            pairs_path=Path(args.pairs) if args.pairs else None,
            per_family=args.per_family,
            seed=args.seed,
            model_name=args.model,
            hook_point=args.hook_point,
            top_k_features=args.top_k_features,
            operation=args.operation,
            factor=args.factor,
            device=args.device,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "out_dir": str(result.out_dir),
                "n_pairs": result.n_pairs,
                "n_features": result.n_features,
                "operation": result.operation,
                "top_features": result.top_features,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
