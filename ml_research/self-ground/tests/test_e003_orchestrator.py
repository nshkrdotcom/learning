from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_e003_calibrated_negation_sae.py"
)
SPEC = importlib.util.spec_from_file_location("run_e003_calibrated_negation_sae", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
e003 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e003)


def _args(tmp_path: Path, *, device: str = "cpu") -> argparse.Namespace:
    return argparse.Namespace(
        device=device,
        task_bank=str(tmp_path / "bank.json"),
        per_family_candidates=80,
        min_calibrated_per_family=10,
        min_baseline_margin=0.1,
        ranking_top_k=50,
        eval_top_k=5,
        operations="ablate",
        random_seeds="7,11,13",
        out_root=str(tmp_path / "runs"),
        force=True,
    )


def test_e003_orchestrator_calls_build_calibrate_rank_eval_inspect_compare(
    tmp_path,
    monkeypatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="{}\n", stderr="")

    monkeypatch.setattr(e003, "_run", fake_run)

    result = e003.run_e003(_args(tmp_path))

    assert result == 0
    joined = [" ".join(command) for command in calls]
    assert any("build_phase3_task_bank.py" in command for command in joined)
    assert any("calibrate_phase3_task_bank.py" in command for command in joined)
    assert any("run_real_activation_ranking.py" in command for command in joined)
    assert any("--task-source file" in command for command in joined)
    assert any("run_negation_ravel_eval.py" in command for command in joined)
    assert any("compare_e002_e003.py" in command for command in joined)


def test_e003_cuda_unavailable_writes_blocker_and_skips_commands(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        e003,
        "_cuda_available",
        lambda: (False, {"cuda_available": False, "blocker": "unit test"}),
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        e003,
        "_run",
        lambda command: calls.append(command)
        or subprocess.CompletedProcess(command, 0, stdout="{}", stderr=""),
    )

    result = e003.run_e003(_args(tmp_path, device="cuda"))

    assert result == 1
    blocker = json.loads(
        (
            tmp_path
            / "runs"
            / "e003_negation_eval_pythia70m_l2_calibrated_pf10_top5_density"
            / "BLOCKED.json"
        ).read_text()
    )
    assert blocker["status"] == "blocked"
    assert "CUDA" in blocker["reason"]
    assert not any("run_real_activation_ranking.py" in " ".join(command) for command in calls)
