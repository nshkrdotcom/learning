from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from self_ground.real_sae_intervention import run_real_sae_intervention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run real decoded SAE feature intervention after semantic SAE metadata, "
            "shape, and reconstruction compatibility checks."
        )
    )
    parser.add_argument("--ranking-dir", default=None)
    parser.add_argument("--pairs", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-family", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--hook-point", default="blocks.2.hook_resid_post")
    parser.add_argument("--sae-release", required=True)
    parser.add_argument("--sae-id", required=True)
    parser.add_argument("--top-k-features", type=int, default=5)
    parser.add_argument("--operation", choices=["ablate", "amplify"], default="ablate")
    parser.add_argument("--factor", type=float, default=1.0)
    parser.add_argument("--patch-mode", choices=["replace", "delta"], default="delta")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.operation == "amplify" and args.factor == 1.0:
        print("ERROR: --operation amplify requires --factor not equal to 1.0", file=sys.stderr)
        return 1
    try:
        result = run_real_sae_intervention(
            out_dir=Path(args.out),
            ranking_dir=Path(args.ranking_dir) if args.ranking_dir else None,
            pairs_path=Path(args.pairs) if args.pairs else None,
            per_family=args.per_family,
            seed=args.seed,
            model_name=args.model,
            hook_point=args.hook_point,
            sae_release=args.sae_release,
            sae_id=args.sae_id,
            top_k_features=args.top_k_features,
            operation=args.operation,
            factor=args.factor,
            patch_mode=args.patch_mode,
            device=args.device,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = {
        "out_dir": str(result.out_dir),
        "n_pairs": result.n_pairs,
        "n_features": result.n_features,
        "operation": result.operation,
        "patch_mode": result.patch_mode,
        "top_features": result.top_features,
        "compatible": result.compatible,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result.compatible else 1


if __name__ == "__main__":
    raise SystemExit(main())
