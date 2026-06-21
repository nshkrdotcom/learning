from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from self_ground.real_ranking import run_activation_ranking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real negation activation ranking.")
    parser.add_argument("--pairs", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-family", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--hook-point", default="blocks.2.hook_resid_post")
    parser.add_argument(
        "--feature-source",
        choices=["residual_dimensions", "sae"],
        default="residual_dimensions",
    )
    parser.add_argument("--pooling", choices=["final_token", "mean"], default="final_token")
    parser.add_argument("--top-k-features", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sae-release", default=None)
    parser.add_argument("--sae-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_activation_ranking(
            out_dir=Path(args.out),
            pairs_path=Path(args.pairs) if args.pairs else None,
            per_family=args.per_family,
            seed=args.seed,
            model_name=args.model,
            hook_point=args.hook_point,
            feature_source=args.feature_source,
            pooling=args.pooling,
            top_k_features=args.top_k_features,
            device=args.device,
            sae_release=args.sae_release,
            sae_id=args.sae_id,
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
                "feature_source": result.feature_source,
                "top_features": result.top_features,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
