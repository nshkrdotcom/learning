from __future__ import annotations

import argparse
import json
from pathlib import Path

from self_ground.sae_compat import verify_sae_compatibility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check real SAELens compatibility.")
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--hook-point", default="blocks.2.hook_resid_post")
    parser.add_argument("--sae-release", required=True)
    parser.add_argument("--sae-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="runs/check_sae_compatibility.json")
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
    )
    print(json.dumps(result.model_dump(), indent=2, sort_keys=True))
    return 0 if result.compatible else 1


if __name__ == "__main__":
    raise SystemExit(main())
