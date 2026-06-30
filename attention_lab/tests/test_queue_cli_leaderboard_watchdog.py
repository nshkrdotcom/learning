from __future__ import annotations

import json
from pathlib import Path

from attention_lab.queue.cli import main as queue_main
from attention_lab.queue.discipline import default_hypothesis_path, validate_hypothesis_doc
from attention_lab.queue.leaderboard import render_leaderboard
from attention_lab.queue.ledger import QueueLedger
from attention_lab.queue.watchdog import Watchdog


def write_hypothesis(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """CLAIM:
This run will lower validation loss versus control.

KILL_CONDITION:
Final val loss is not at least 0.02 lower than control.

MECHANISM_PROOF:
cp_gradient_norm > 1e-4 by step 500.

NEAREST_BORING_EXPLANATION:
Extra low-rank parameters.

CONTROL_THAT_RULES_IT_OUT:
cp_bilinear_r8_30m_seed1.
""",
        encoding="utf-8",
    )


def test_hypothesis_doc_validation_and_default_path(tmp_path, tiny_config):
    config = tiny_config(tmp_path, tmp_path / "data")
    config_path = Path("configs/experiments/E999_example/candidate.yaml")
    assert default_hypothesis_path(config_path, config) == Path("docs/experiments/E999_example/hypothesis_tiny_test_run.md")

    hypothesis_path = tmp_path / "hypothesis.md"
    write_hypothesis(hypothesis_path)
    result = validate_hypothesis_doc(hypothesis_path)
    assert result.ok is True
    assert result.missing_fields == []


def test_leaderboard_renders_mechanism_and_missing_values():
    rows = [
        {
            "config_name": "cp_trilinear_r8_30m_seed1",
            "attention_type": "cp_trilinear",
            "stage": "FULL",
            "status": "PASSED",
            "final_val_loss": 4.0,
            "final_ppl": 54.6,
            "median_tokens_per_sec": 100000.0,
            "peak_vram_allocated_mb": 3200.0,
            "mechanism_active": 1,
            "notes": "SHOWS: active",
        },
        {
            "config_name": "candidate_pending",
            "attention_type": "cp_bilinear",
            "stage": "SCREEN",
            "status": "PENDING",
            "final_val_loss": None,
            "final_ppl": None,
            "median_tokens_per_sec": None,
            "peak_vram_allocated_mb": None,
            "mechanism_active": None,
            "notes": "",
        },
    ]
    output = render_leaderboard(rows, now="2026-06-30 07:14")
    assert "QUEUE STATUS  2026-06-30 07:14" in output
    assert "cp_trilinear_r8_30m_seed1" in output
    assert "4.000" in output
    assert "100k" in output
    assert "candidate_pending" in output
    assert "---" in output


def test_cli_ls_note_requeue_and_status(tmp_path, tiny_config, capsys):
    db_path = tmp_path / "queue.db"
    ledger = QueueLedger(db_path)
    ledger.initialize()
    config = tiny_config(tmp_path, tmp_path / "data")
    config_path = tmp_path / "candidate.yaml"
    import yaml

    content = yaml.safe_dump(config).encode()
    config_path.write_bytes(content)
    run_id = ledger.enqueue_config(config_path, config, content)
    ledger.promote_to_full(run_id)
    ledger.mark_failed(run_id, failure_class="NAN")
    ledger.close()

    queue_main(["--db", str(db_path), "ls"])
    assert "candidate" in capsys.readouterr().out

    queue_main(["--db", str(db_path), "note", run_id, "SHOWS: failed with NAN"])
    queue_main(["--db", str(db_path), "show", run_id])
    assert "SHOWS: failed with NAN" in capsys.readouterr().out

    queue_main(["--db", str(db_path), "requeue", run_id])
    queue_main(["--db", str(db_path), "status"])
    output = capsys.readouterr().out
    assert "QUEUE STATUS" in output
    assert "PENDING" in output


def test_watchdog_screens_before_full_and_skips_missing_hypothesis(tmp_path, tiny_config):
    db_path = tmp_path / "queue.db"
    ledger = QueueLedger(db_path)
    ledger.initialize()
    config = tiny_config(tmp_path, tmp_path / "data")
    config_path = tmp_path / "candidate.yaml"
    import yaml

    content = yaml.safe_dump(config).encode()
    config_path.write_bytes(content)
    run_id = ledger.enqueue_config(config_path, config, content)

    events = []

    def screen_runner(row, ledger_obj):
        events.append(("screen", row["id"]))
        ledger_obj.promote_to_full(row["id"])
        return {"ok": True}

    def full_runner(row, ledger_obj):
        events.append(("full", row["id"]))
        ledger_obj.mark_passed(row["id"])
        return {"ok": True}

    watchdog = Watchdog(ledger, screen_runner=screen_runner, full_runner=full_runner, sleep_seconds=0)
    watchdog.run_once()
    assert events == [("screen", run_id)]

    # The promoted run has no hypothesis doc, so the full runner is skipped and annotated.
    watchdog.run_once()
    assert events == [("screen", run_id)]
    assert "hypothesis" in (ledger.get_run(run_id)["notes"] or "").lower()

    hypothesis_path = default_hypothesis_path(config_path, config)
    write_hypothesis(hypothesis_path)
    try:
        watchdog.run_once()
        assert events == [("screen", run_id), ("full", run_id)]
    finally:
        hypothesis_path.unlink(missing_ok=True)
        try:
            hypothesis_path.parent.rmdir()
        except OSError:
            pass


def test_cli_add_validates_and_copies(tmp_path, tiny_config, capsys):
    root = tmp_path / "root"
    root.mkdir()
    db_path = root / "data" / "queue.db"
    source = tmp_path / "candidate.yaml"
    import yaml

    source.write_text(yaml.safe_dump(tiny_config(tmp_path, tmp_path / "data")), encoding="utf-8")
    queue_main(["--root", str(root), "--db", str(db_path), "add", str(source)])
    output = capsys.readouterr().out
    assert "added:" in output
    assert (root / "queue" / "inbox" / "candidate.yaml").exists()
    assert json.dumps({"ok": True}) != ""
