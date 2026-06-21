from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from self_ground.real_model_check import check_real_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check real TransformerLens activation capture.")
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--hook-point", default="blocks.2.hook_resid_post")
    parser.add_argument("--out", default="runs/check_real_model.json")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        artifact = check_real_model(
            model_name=args.model,
            hook_point=args.hook_point,
            out=Path(args.out),
            device=args.device,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
