from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

MODEL = "EleutherAI/pythia-70m-deduped"
HOOK_POINT = "blocks.2.hook_resid_post"
SAE_RELEASE = "pythia-70m-deduped-res-sm"
SAE_ID = "blocks.2.hook_resid_post"
TASK_SOURCE_ID = "e003_pythia70m_deduped_baseline_calibrated_bank_v1"
E002_DIR = Path("runs/e002_negation_ravel_eval_pythia70m_deduped_l2_pf10_top5_density")


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _cuda_available() -> tuple[bool, dict[str, Any]]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        return available, {
            "torch_available": True,
            "torch_version": getattr(torch, "__version__", None),
            "cuda_available": available,
            "cuda_device_count": int(torch.cuda.device_count()) if available else 0,
            "cuda_device_names": [
                torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())
            ]
            if available
            else [],
        }
    except Exception as exc:
        return False, {
            "torch_available": False,
            "cuda_available": False,
            "exception_class": type(exc).__name__,
            "exception_message": str(exc),
        }


def _write_blocker(eval_dir: Path, *, reason: str, details: dict[str, Any]) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "blocked",
        "blocker_type": "e003_capability_or_calibration_blocker",
        "reason": reason,
        "details": details,
        "no_intervention_rows_written": True,
    }
    _json_dump(eval_dir / "BLOCKED.json", payload)
    (eval_dir / "README.md").write_text(
        f"""# E003 Blocked

- reason: `{reason}`

No decoded SAE intervention rows were written for this blocked E003 run.
The blocker is artifact-backed in `BLOCKED.json`.
""",
        encoding="utf-8",
    )


def _command_log(out_dir: Path, commands: list[list[str]]) -> None:
    _json_dump(
        out_dir / "e003_command_log.json",
        {"commands": [" ".join(command) for command in commands]},
    )


def run_e003(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root)
    safe_margin = str(args.min_baseline_margin).replace(".", "p")
    calibration_dir = out_root / (
        f"e003_task_bank_calibration_pythia70m_margin{safe_margin}_"
        f"min{args.min_calibrated_per_family}"
    )
    ranking_dir = out_root / (
        f"e003_real_sae_ranking_pythia70m_l2_calibrated_"
        f"pf{args.min_calibrated_per_family}_top{args.ranking_top_k}"
    )
    eval_dir = out_root / (
        f"e003_negation_eval_pythia70m_l2_calibrated_"
        f"pf{args.min_calibrated_per_family}_top{args.eval_top_k}_density"
    )
    comparison_dir = out_root / "diagnostics" / "e003_vs_e002_comparison"
    feature_diagnostics_dir = out_root / "diagnostics" / "e003_feature_selection"
    task_bank_path = Path(args.task_bank)
    calibrated_task_file = calibration_dir / "calibrated_behavioral_tasks.jsonl"
    stale_eval_blocker = eval_dir / "BLOCKED.json"
    if stale_eval_blocker.exists():
        stale_eval_blocker.unlink()

    if args.device.startswith("cuda"):
        cuda_ok, capability = _cuda_available()
        _json_dump(out_root / "e003_capability_check.json", capability)
        if not cuda_ok:
            _write_blocker(
                eval_dir,
                reason="CUDA was requested for serious E003 but is unavailable.",
                details=capability,
            )
            _run(
                [
                    sys.executable,
                    "scripts/compare_e002_e003.py",
                    "--e002",
                    str(E002_DIR),
                    "--e003",
                    str(eval_dir),
                    "--out",
                    str(comparison_dir),
                ]
            )
            return 1

    commands: list[list[str]] = []
    if args.force or not task_bank_path.exists():
        commands.append(
            [
                sys.executable,
                "scripts/build_phase3_task_bank.py",
                "--model",
                MODEL,
                "--out",
                str(task_bank_path),
                "--per-family-candidates",
                str(args.per_family_candidates),
                "--device",
                args.device,
            ]
        )
    if args.force or not calibrated_task_file.exists():
        commands.append(
            [
                sys.executable,
                "scripts/calibrate_phase3_task_bank.py",
                "--task-bank",
                str(task_bank_path),
                "--model",
                MODEL,
                "--device",
                args.device,
                "--out",
                str(calibration_dir),
                "--min-baseline-margin",
                str(args.min_baseline_margin),
                "--min-per-family",
                str(args.min_calibrated_per_family),
            ]
        )
    commands.append(
        [
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
            args.device,
            "--per-family",
            str(args.min_calibrated_per_family),
            "--top-k-features",
            str(args.ranking_top_k),
            "--out",
            str(ranking_dir),
            "--task-source",
            "file",
            "--task-file",
            str(calibrated_task_file),
            "--task-source-id",
            TASK_SOURCE_ID,
        ]
    )
    commands.append(
        [
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
            str(args.min_calibrated_per_family),
            "--top-k-features",
            str(args.eval_top_k),
            "--baseline-mode",
            "top-vs-random-density-and-bottom-active",
            "--random-seeds",
            args.random_seeds,
            "--operations",
            args.operations,
            "--patch-mode",
            "delta",
            "--device",
            args.device,
            "--task-source",
            "file",
            "--task-file",
            str(calibrated_task_file),
            "--task-bank-calibration-dir",
            str(calibration_dir),
            "--task-source-id",
            TASK_SOURCE_ID,
        ]
    )
    commands.append(
        [
            sys.executable,
            "scripts/analyze_feature_selection.py",
            "--ranking-dir",
            str(ranking_dir),
            "--eval-dir",
            str(eval_dir),
            "--out",
            str(feature_diagnostics_dir),
        ]
    )
    commands.append(
        [
            sys.executable,
            "scripts/compare_e002_e003.py",
            "--e002",
            str(E002_DIR),
            "--e003",
            str(eval_dir),
            "--out",
            str(comparison_dir),
        ]
    )
    _command_log(eval_dir, commands)

    for command in commands:
        completed = _run(command)
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        if completed.returncode != 0:
            _write_blocker(
                eval_dir,
                reason=f"E003 command failed: {' '.join(command)}",
                details={
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
            )
            return completed.returncode

    inspection = _run(
        [
            sys.executable,
            "scripts/inspect_claim_run.py",
            "--run-dir",
            str(eval_dir),
            "--json",
        ]
    )
    if inspection.returncode == 0:
        (eval_dir / "inspection_summary.json").write_text(
            inspection.stdout,
            encoding="utf-8",
        )
        print(inspection.stdout, end="")
    else:
        print(inspection.stderr, file=sys.stderr, end="")
        return inspection.returncode

    readme = f"""# E003 Calibrated Negation SAE Run

- model: `{MODEL}`
- hook point: `{HOOK_POINT}`
- SAE release: `{SAE_RELEASE}`
- SAE id: `{SAE_ID}`
- task bank: `{task_bank_path}`
- calibration dir: `{calibration_dir}`
- ranking dir: `{ranking_dir}`
- evaluation dir: `{eval_dir}`
- comparison dir: `{comparison_dir}`

This run uses a baseline-calibrated task source. Any candidate evidence is
conditional on the task-bank calibration artifacts and the current custom
token-contrast evaluator.
"""
    (eval_dir / "README.md").write_text(readme, encoding="utf-8")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E003 calibrated negation SAE experiment.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--task-bank",
        default="data/phase3_task_bank/pythia70m_negation_candidate_bank.json",
    )
    parser.add_argument("--per-family-candidates", type=int, default=80)
    parser.add_argument("--min-calibrated-per-family", type=int, default=10)
    parser.add_argument("--min-baseline-margin", type=float, default=0.1)
    parser.add_argument("--ranking-top-k", type=int, default=50)
    parser.add_argument("--eval-top-k", type=int, default=5)
    parser.add_argument("--operations", default="ablate")
    parser.add_argument("--random-seeds", default="7,11,13")
    parser.add_argument("--out-root", default="runs")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    return run_e003(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
