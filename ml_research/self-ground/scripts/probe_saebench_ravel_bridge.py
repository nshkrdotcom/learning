from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from self_ground.ravel_adapter.saebench_probe import probe_saebench_ravel_bridge


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe whether installed SAEBench/RAVEL APIs can support SELF-GROUND."
    )
    parser.add_argument("--out", default="runs/probe_saebench_ravel_bridge")
    args = parser.parse_args()

    try:
        result = probe_saebench_ravel_bridge(out_dir=Path(args.out))
    except Exception as exc:
        print(f"SAEBench/RAVEL probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
