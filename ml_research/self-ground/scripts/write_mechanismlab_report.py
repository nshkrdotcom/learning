from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from self_ground.mechanismlab_adapter import write_mechanismlab_artifacts_for_phase3


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write generic MechanismLab artifacts for an existing SELF-GROUND Phase 3 run."
    )
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    try:
        report = write_mechanismlab_artifacts_for_phase3(Path(args.run_dir))
    except Exception as exc:
        print(f"failed to write MechanismLab report: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
