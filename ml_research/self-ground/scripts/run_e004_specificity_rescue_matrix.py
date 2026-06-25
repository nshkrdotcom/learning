from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

MODEL = "EleutherAI/pythia-70m-deduped"
SAE_RELEASE = "pythia-70m-deduped-res-sm"
TASK_SOURCE_ID = "e003_calibrated_task_bank_for_e004"
BASELINE_MODE = "top-vs-random-density-and-bottom-active"
PATCH_MODE = "delta"


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _layer_slug(hook_point: str) -> str:
    parts = hook_point.split(".")
    if len(parts) >= 2 and parts[0] == "blocks":
        return f"block{parts[1]}"
    return hook_point.replace(".", "_")


def _cell_slug(layer: str, mode: str, operations: str, control_suite: str) -> str:
    return "_".join(
        [
            _layer_slug(layer),
            mode.replace("top-", "").replace("-", "_"),
            operations.replace(",", "_"),
            control_suite.replace("_control", "").replace("_", ""),
        ]
    )


def _write_blocker(path: Path, *, reason: str, command: list[str] | None, result=None) -> None:
    payload = {
        "status": "blocked",
        "reason": reason,
        "command": " ".join(command) if command else None,
        "returncode": getattr(result, "returncode", None),
        "stdout": getattr(result, "stdout", None),
        "stderr": getattr(result, "stderr", None),
        "no_fake_rows_written": True,
    }
    _write_json(path / "CELL_BLOCKED.json", payload)
    (path / "README.md").write_text(
        f"""# E004 Matrix Cell Blocked

- reason: `{reason}`

See `CELL_BLOCKED.json` for command output. This blocked cell does not support
candidate evidence.
""",
        encoding="utf-8",
    )


def _eval_cell_complete(path: Path) -> bool:
    return (path / "mechanism_report.json").exists() and (
        path / "inspection_summary.json"
    ).exists()


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    layers = _split_csv(args.layers)
    modes = _split_csv(args.feature_selection_modes)
    operations = args.operations
    attempted = 0
    completed = 0
    blocked = 0
    cells: list[dict[str, Any]] = []

    if args.device.startswith("cuda"):
        cuda_ok, capability = _cuda_available()
        _write_json(out_root / "capability.json", capability)
        if not cuda_ok:
            blocker = {
                "status": "blocked",
                "reason": "CUDA was requested for E004 but is unavailable.",
                "capability": capability,
                "attempted_cells": 0,
            }
            _write_json(out_root / "BLOCKED.json", blocker)
            return blocker

    for layer in layers:
        layer_slug = _layer_slug(layer)
        ranking_dir = (
            out_root
            / "rankings"
            / f"{layer_slug}_rich_calibrated_top{args.ranking_top_k}"
        )
        layer_eval_dirs = {
            mode: out_root / "eval" / _cell_slug(layer, mode, operations, args.control_suite)
            for mode in modes
        }
        layer_eval_complete = (
            all(_eval_cell_complete(path) for path in layer_eval_dirs.values())
            and not args.force
        )
        ranking_command = [
            sys.executable,
            "scripts/run_real_activation_ranking.py",
            "--model",
            MODEL,
            "--hook-point",
            layer,
            "--feature-source",
            "sae",
            "--sae-release",
            SAE_RELEASE,
            "--sae-id",
            layer,
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
            str(args.task_file),
            "--task-source-id",
            TASK_SOURCE_ID,
        ]
        ranking_ok = layer_eval_complete or (
            (ranking_dir / "feature_rankings.csv").exists() and not args.force
        )
        ranking_result = None
        if not ranking_ok:
            ranking_result = _run(ranking_command)
            ranking_ok = ranking_result.returncode == 0
            if not ranking_ok:
                _write_blocker(
                    ranking_dir,
                    reason="ranking failed for layer",
                    command=ranking_command,
                    result=ranking_result,
                )

        for mode in modes:
            attempted += 1
            eval_dir = out_root / "eval" / _cell_slug(layer, mode, operations, args.control_suite)
            if _eval_cell_complete(eval_dir) and not args.force:
                completed += 1
                cells.append(
                    {
                        "layer": layer,
                        "feature_selection_mode": mode,
                        "operation": operations,
                        "control_suite": args.control_suite,
                        "status": "completed",
                        "run_dir": str(eval_dir),
                        "resumed_from_existing_artifacts": True,
                    }
                )
                continue
            if not ranking_ok:
                blocked += 1
                _write_blocker(
                    eval_dir,
                    reason="ranking failed for this layer; evaluation not run",
                    command=ranking_command,
                    result=ranking_result,
                )
                cells.append(
                    {
                        "layer": layer,
                        "feature_selection_mode": mode,
                        "operation": operations,
                        "control_suite": args.control_suite,
                        "status": "blocked",
                        "run_dir": str(eval_dir),
                        "blocker": "ranking_failed",
                    }
                )
                continue
            eval_command = [
                sys.executable,
                "scripts/run_negation_ravel_eval.py",
                "--ranking-dir",
                str(ranking_dir),
                "--out",
                str(eval_dir),
                "--model",
                MODEL,
                "--hook-point",
                layer,
                "--sae-release",
                SAE_RELEASE,
                "--sae-id",
                layer,
                "--per-family",
                str(args.min_calibrated_per_family),
                "--top-k-features",
                str(args.eval_top_k),
                "--baseline-mode",
                BASELINE_MODE,
                "--random-seeds",
                args.random_seeds,
                "--operations",
                operations,
                "--amplify-factors",
                args.amplify_factors,
                "--patch-mode",
                PATCH_MODE,
                "--device",
                args.device,
                "--task-source",
                "file",
                "--task-file",
                str(args.task_file),
                "--task-bank-calibration-dir",
                str(args.task_bank_calibration_dir),
                "--task-source-id",
                TASK_SOURCE_ID,
                "--feature-selection-mode",
                mode,
                "--min-family-consistency",
                str(args.min_family_consistency),
                "--control-suite",
                args.control_suite,
            ]
            result = _run(eval_command)
            if result.returncode != 0:
                blocked += 1
                _write_blocker(
                    eval_dir,
                    reason="evaluation failed for matrix cell",
                    command=eval_command,
                    result=result,
                )
                status = "blocked"
            else:
                inspect_command = [
                    sys.executable,
                    "scripts/inspect_claim_run.py",
                    "--run-dir",
                    str(eval_dir),
                    "--json",
                ]
                inspect_result = _run(inspect_command)
                if inspect_result.returncode == 0:
                    (eval_dir / "inspection_summary.json").write_text(
                        inspect_result.stdout,
                        encoding="utf-8",
                    )
                completed += 1
                status = "completed"
            cells.append(
                {
                    "layer": layer,
                    "feature_selection_mode": mode,
                    "operation": operations,
                    "control_suite": args.control_suite,
                    "status": status,
                    "run_dir": str(eval_dir),
                }
            )

    summary = {
        "status": "ok",
        "attempted_cells": attempted,
        "completed_cells": completed,
        "blocked_cells": blocked,
        "layers": layers,
        "feature_selection_modes": modes,
        "operations": operations,
        "control_suite": args.control_suite,
        "cells": cells,
    }
    _write_json(out_root / "matrix_run_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the bounded E004 specificity rescue matrix.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-file", required=True, type=Path)
    parser.add_argument("--task-bank-calibration-dir", required=True, type=Path)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--feature-selection-modes", required=True)
    parser.add_argument("--operations", default="ablate,amplify")
    parser.add_argument("--amplify-factors", default="2.0")
    parser.add_argument("--control-suite", default="multi_control")
    parser.add_argument("--ranking-top-k", type=int, default=100)
    parser.add_argument("--eval-top-k", type=int, default=5)
    parser.add_argument("--min-calibrated-per-family", type=int, default=10)
    parser.add_argument("--min-family-consistency", type=int, default=2)
    parser.add_argument("--random-seeds", default="7,11,13")
    parser.add_argument("--out-root", default="runs/e004_specificity_rescue_matrix")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_matrix(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary.get("status") == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
