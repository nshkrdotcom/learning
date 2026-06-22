from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from self_ground.task_bank import build_candidate_task_bank, write_task_bank_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a tokenizer-filtered Phase 3 task bank.")
    parser.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-family-candidates", type=int, default=80)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        from self_ground.model import TransformerLensModelAdapter

        model_adapter = TransformerLensModelAdapter(model_name=args.model, device=args.device)
        bank, tasks, rejections = build_candidate_task_bank(
            model_adapter=model_adapter,
            model_name=args.model,
            per_family_candidates=args.per_family_candidates,
        )
        write_task_bank_artifacts(
            bank=bank,
            tasks=tasks,
            rejections=rejections,
            out=Path(args.out),
        )
    except Exception as exc:
        print(f"task bank build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "out": str(Path(args.out)),
                "accepted_templates": len(bank.templates),
                "accepted_by_family": bank.metadata.get("accepted_by_family"),
                "rejected_count": len(rejections),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
