from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from self_ground.io import write_config

try:
    from scripts.check_run_capability import collect_run_capability
except ModuleNotFoundError:  # direct execution as scripts/run_e002_*.py
    from check_run_capability import collect_run_capability

MODEL = "EleutherAI/pythia-70m-deduped"
HOOK_POINT = "blocks.2.hook_resid_post"
SAE_RELEASE = "pythia-70m-deduped-res-sm"
SAE_ID = "blocks.2.hook_resid_post"
BASELINE_MODE = "top-vs-random-density-and-bottom-active"
PATCH_MODE = "delta"


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _run_slug(
    *,
    device: str,
    ranking_per_family: int,
    eval_per_family: int,
    ranking_top_k: int,
    eval_top_k: int,
) -> tuple[str, str]:
    prefix = "e002" if device.startswith("cuda") else "e002_cpu_diagnostic"
    ranking = (
        f"{prefix}_real_sae_ranking_pythia70m_deduped_l2_pf"
        f"{ranking_per_family}_top{ranking_top_k}"
    )
    eval_name = (
        f"{prefix}_negation_ravel_eval_pythia70m_deduped_l2_pf{eval_per_family}"
        f"_top{eval_top_k}_density"
    )
    return ranking, eval_name


def _write_blocked(
    *,
    eval_dir: Path,
    capability: dict[str, Any],
    args: argparse.Namespace,
    reason: str,
) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "blocked",
        "reason": reason,
        "capability": capability,
        "requested_device": args.device,
        "allow_cpu_serious_run": args.allow_cpu_serious_run,
        "no_ranking_or_evaluation_commands_run": True,
    }
    write_config(payload, eval_dir / "BLOCKED.json")
    readme = f"""# E002 Blocked

The serious E002 run was not started.

- reason: `{reason}`
- requested device: `{args.device}`
- can attempt GPU: `{capability.get("can_attempt_e002_gpu")}`

See `BLOCKED.json` for exact capability blockers.
"""
    (eval_dir / "README.md").write_text(readme, encoding="utf-8")


def _run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _jsonable(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, default=str))


def run_e002(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    ranking_name, eval_name = _run_slug(
        device=args.device,
        ranking_per_family=args.ranking_per_family,
        eval_per_family=args.eval_per_family,
        ranking_top_k=args.ranking_top_k,
        eval_top_k=args.eval_top_k,
    )
    ranking_dir = out_root / ranking_name
    eval_dir = out_root / eval_name
    capability_dir = out_root / "capability_check"
    capability = collect_run_capability(
        out_dir=capability_dir,
        model=MODEL,
        sae_release=SAE_RELEASE,
        sae_id=SAE_ID,
    )
    capability = _jsonable(capability)
    actual_device = args.device
    if args.device.startswith("cuda") and not capability["can_attempt_e002_gpu"]:
        if not args.allow_cpu_serious_run:
            _write_blocked(
                eval_dir=eval_dir,
                capability=capability,
                args=args,
                reason="CUDA requested for serious E002 but capability check failed",
            )
            return {
                "status": "blocked",
                "ranking_dir": str(ranking_dir),
                "eval_dir": str(eval_dir),
                "capability_dir": str(capability_dir),
            }
        actual_device = "cpu"
    if actual_device == "cpu" and not args.allow_cpu_serious_run:
        _write_blocked(
            eval_dir=eval_dir,
            capability=capability,
            args=args,
            reason="CPU E002 execution requires --allow-cpu-serious-run true",
        )
        return {
            "status": "blocked",
            "ranking_dir": str(ranking_dir),
            "eval_dir": str(eval_dir),
            "capability_dir": str(capability_dir),
        }

    ranking_cmd = [
        sys.executable,
        "scripts/run_real_activation_ranking.py",
        "--model",
        MODEL,
        "--hook-point",
        HOOK_POINT,
        "--feature-source",
        "sae",
        "--sae-release",
        SAE_RELEASE,
        "--sae-id",
        SAE_ID,
        "--device",
        actual_device,
        "--per-family",
        str(args.ranking_per_family),
        "--top-k-features",
        str(args.ranking_top_k),
        "--out",
        str(ranking_dir),
    ]
    eval_cmd = [
        sys.executable,
        "scripts/run_negation_ravel_eval.py",
        "--ranking-dir",
        str(ranking_dir),
        "--out",
        str(eval_dir),
        "--model",
        MODEL,
        "--hook-point",
        HOOK_POINT,
        "--sae-release",
        SAE_RELEASE,
        "--sae-id",
        SAE_ID,
        "--per-family",
        str(args.eval_per_family),
        "--top-k-features",
        str(args.eval_top_k),
        "--baseline-mode",
        BASELINE_MODE,
        "--random-seeds",
        args.random_seeds,
        "--operations",
        args.operations,
        "--patch-mode",
        PATCH_MODE,
        "--device",
        actual_device,
    ]
    inspect_cmd = [
        sys.executable,
        "scripts/inspect_claim_run.py",
        "--run-dir",
        str(eval_dir),
        "--json",
    ]
    if args.force or not (ranking_dir / "feature_rankings.csv").exists():
        _run_command(ranking_cmd)
    _run_command(eval_cmd)
    inspection = subprocess.run(inspect_cmd, check=True, text=True, capture_output=True)
    inspection_payload = json.loads(inspection.stdout)
    write_config(inspection_payload, eval_dir / "inspection_summary.json")
    readme = f"""# E002 SELF-GROUND Run

- actual device: `{actual_device}`
- ranking command: `{" ".join(ranking_cmd)}`
- evaluation command: `{" ".join(eval_cmd)}`
- claim status: `{inspection_payload.get("claim", {}).get("claim_status")}`
- run classification: `{inspection_payload.get("run_classification")}`

Local JSONL/CSV/Markdown artifacts remain the source of truth. This wrapper
only orchestrates existing SELF-GROUND scripts.
"""
    (eval_dir / "README_E002.md").write_text(readme, encoding="utf-8")
    return {
        "status": "completed",
        "actual_device": actual_device,
        "ranking_dir": str(ranking_dir),
        "eval_dir": str(eval_dir),
        "capability_dir": str(capability_dir),
        "inspection_summary": str(eval_dir / "inspection_summary.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or gate the E002 negation SAE run.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ranking-per-family", type=int, default=10)
    parser.add_argument("--eval-per-family", type=int, default=10)
    parser.add_argument("--ranking-top-k", type=int, default=50)
    parser.add_argument("--eval-top-k", type=int, default=5)
    parser.add_argument("--operations", default="ablate")
    parser.add_argument("--random-seeds", default="7,11,13")
    parser.add_argument("--out-root", default="runs")
    parser.add_argument("--allow-cpu-serious-run", type=_bool_arg, default=False)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run_e002(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"completed", "blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
