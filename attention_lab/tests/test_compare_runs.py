from __future__ import annotations

import json

from attention_lab.training.compare_runs import compare_runs


def _write_summary(run_dir, run_name: str, loss: float) -> None:
    evals_dir = run_dir / "evals"
    evals_dir.mkdir(parents=True)
    summary = {
        "run_dir": str(run_dir),
        "run_name": run_name,
        "max_step": 2,
        "train_event_count": 2,
        "val_event_count": 2,
        "initial_val_loss": 2.0,
        "final_val_loss": loss,
        "best_val_loss": loss,
        "initial_val_perplexity": 7.0,
        "final_val_perplexity": 3.0,
        "median_tokens_per_sec": 10.0,
        "peak_vram_mb": 5.0,
        "checkpoint_count": 1,
    }
    (evals_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_compare_runs_reads_summary_contract(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_summary(baseline, "baseline", 1.0)
    _write_summary(candidate, "candidate", 0.9)

    rows = compare_runs(baseline, candidate)

    assert [row["role"] for row in rows] == ["baseline", "candidate"]
    assert rows[0]["run_name"] == "baseline"
    assert rows[1]["final_val_loss"] == 0.9
    assert rows[1]["peak_vram_allocated_mb"] == 5.0
