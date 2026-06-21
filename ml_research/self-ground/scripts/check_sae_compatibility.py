from __future__ import annotations

import argparse
import json
from pathlib import Path

from self_ground.sae_compat import verify_sae_compatibility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check real SAELens semantic compatibility. Shape-only diagnostic "
            "output is not production compatibility."
        )
    )
    parser.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--hook-point", default="blocks.2.hook_resid_post")
    parser.add_argument("--sae-release", required=True)
    parser.add_argument("--sae-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="runs/check_sae_compatibility.json")
    parser.add_argument(
        "--require-metadata-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require SAE-declared model and hook metadata to match the requested target.",
    )
    parser.add_argument(
        "--allow-shape-only-diagnostic",
        action="store_true",
        help="Compute shape-only diagnostic fields; not production-compatible.",
    )
    parser.add_argument("--max-reconstruction-l2-relative", type=float, default=None)
    parser.add_argument("--max-reconstruction-mse", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = verify_sae_compatibility(
        model_name=args.model,
        hook_point=args.hook_point,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        device=args.device,
        out=Path(args.out),
        require_metadata_match=args.require_metadata_match,
        allow_shape_only_diagnostic=args.allow_shape_only_diagnostic,
        max_reconstruction_l2_relative=args.max_reconstruction_l2_relative,
        max_reconstruction_mse=args.max_reconstruction_mse,
    )
    print(json.dumps(result.model_dump(), indent=2, sort_keys=True))
    return 0 if result.compatible else 1


if __name__ == "__main__":
    raise SystemExit(main())
