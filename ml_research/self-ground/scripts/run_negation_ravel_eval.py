from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from self_ground.real_behavioral_intervention import run_real_behavioral_sae_intervention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the SELF-GROUND negation RAVEL-style token-contrast evaluation. "
            "This wraps the existing TransformerLens + SAELens decoded intervention "
            "path and writes the claim-ledger artifacts."
        )
    )
    parser.add_argument("--ranking-dir", required=True)
    parser.add_argument("--out", default="runs/negation_ravel_eval")
    parser.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--hook-point", default="blocks.2.hook_resid_post")
    parser.add_argument("--sae-release", required=True)
    parser.add_argument("--sae-id", required=True)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--per-family", type=int, default=2)
    parser.add_argument("--top-k-features", type=int, default=2)
    parser.add_argument("--baseline-mode", default="top-vs-random-multiseed")
    parser.add_argument("--random-seeds", default="7,11,13")
    parser.add_argument("--operations", default="ablate")
    parser.add_argument("--amplify-factors", default="2.0")
    parser.add_argument("--patch-mode", default="delta", choices=["replace", "delta"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--density-tolerance", type=float, default=0.10)
    parser.add_argument("--abs-mean-tolerance", type=float, default=0.10)
    parser.add_argument(
        "--allow-relaxed-density-matching",
        dest="allow_relaxed_density_matching",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-allow-relaxed-density-matching",
        dest="allow_relaxed_density_matching",
        action="store_false",
    )
    parser.add_argument(
        "--allow-metadata-mismatch",
        action="store_true",
        help="Diagnostic only; cannot produce candidate evidence.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip mechanism_report.json/md writing.",
    )
    return parser.parse_args()


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_csv_strings(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    try:
        result = run_real_behavioral_sae_intervention(
            ranking_dir=Path(args.ranking_dir),
            out_dir=Path(args.out),
            tasks_path=Path(args.tasks) if args.tasks else None,
            per_family=args.per_family,
            model_name=args.model,
            hook_point=args.hook_point,
            sae_release=args.sae_release,
            sae_id=args.sae_id,
            top_k_features=args.top_k_features,
            baseline_mode=args.baseline_mode,
            random_seeds=_parse_csv_ints(args.random_seeds),
            operations=_parse_csv_strings(args.operations),
            amplify_factors=_parse_csv_floats(args.amplify_factors),
            patch_mode=args.patch_mode,
            device=args.device,
            allow_metadata_mismatch=args.allow_metadata_mismatch,
            write_report=not args.no_report,
            density_tolerance=args.density_tolerance,
            abs_mean_tolerance=args.abs_mean_tolerance,
            allow_relaxed_density_matching=args.allow_relaxed_density_matching,
        )
    except Exception as exc:
        print(
            f"negation RAVEL-style evaluation failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            {
                "out_dir": str(result.out_dir),
                "n_tasks_total": result.n_tasks_total,
                "n_tasks_valid": result.n_tasks_valid,
                "n_feature_sets": result.n_feature_sets,
                "n_rows": result.n_rows,
                "compatible": result.compatible,
                "report_written": result.report_written,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
