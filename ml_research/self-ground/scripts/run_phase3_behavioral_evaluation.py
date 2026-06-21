from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from self_ground.real_behavioral_intervention import (
    parse_amplify_factors,
    parse_int_list,
    parse_operations,
    run_real_behavioral_sae_intervention,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 3 token-contrast evaluation for real decoded SAE interventions. "
            "Requires a prior SAE ranking directory."
        )
    )
    parser.add_argument("--ranking-dir", required=True)
    parser.add_argument("--tasks")
    parser.add_argument("--out", default="runs/test_phase3_behavioral_evaluation")
    parser.add_argument("--per-family", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--hook-point", default="blocks.2.hook_resid_post")
    parser.add_argument("--sae-release", required=True)
    parser.add_argument("--sae-id", required=True)
    parser.add_argument("--top-k-features", type=int, default=5)
    parser.add_argument("--baseline-mode", default="top-vs-random-multiseed")
    parser.add_argument("--random-seeds", default="7,11,13")
    parser.add_argument("--operations", default="ablate")
    parser.add_argument("--amplify-factors", default="2.0")
    parser.add_argument("--patch-mode", default="delta")
    parser.add_argument("--token-position", type=int, default=-1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reduction", default="mean")
    parser.add_argument("--min-valid-tasks-per-family", type=int, default=2)
    parser.add_argument("--allow-metadata-mismatch", action="store_true")
    parser.add_argument("--write-report", dest="write_report", action="store_true", default=True)
    parser.add_argument("--no-write-report", dest="write_report", action="store_false")
    args = parser.parse_args()

    try:
        result = run_real_behavioral_sae_intervention(
            out_dir=Path(args.out),
            ranking_dir=Path(args.ranking_dir),
            tasks_path=Path(args.tasks) if args.tasks else None,
            per_family=args.per_family,
            seed=args.seed,
            model_name=args.model,
            hook_point=args.hook_point,
            sae_release=args.sae_release,
            sae_id=args.sae_id,
            top_k_features=args.top_k_features,
            baseline_mode=args.baseline_mode,
            random_seeds=parse_int_list(args.random_seeds),
            operations=parse_operations(args.operations),
            amplify_factors=parse_amplify_factors(args.amplify_factors),
            patch_mode=args.patch_mode,
            token_position=args.token_position,
            device=args.device,
            reduction=args.reduction,
            min_valid_tasks_per_family=args.min_valid_tasks_per_family,
            allow_metadata_mismatch=args.allow_metadata_mismatch,
            write_report=args.write_report,
        )
    except Exception as exc:
        print(f"Phase 3 token-contrast evaluation failed: {exc}", file=sys.stderr)
        return 1

    payload = {
        "out_dir": str(result.out_dir),
        "n_tasks_total": result.n_tasks_total,
        "n_tasks_valid": result.n_tasks_valid,
        "n_tasks_excluded": result.n_tasks_excluded,
        "n_feature_sets": result.n_feature_sets,
        "n_rows": result.n_rows,
        "compatible": result.compatible,
        "task_validation_passed": result.task_validation_passed,
        "report_written": result.report_written,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.compatible and result.task_validation_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
