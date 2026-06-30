from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
from pathlib import Path

from attention_lab.queue.leaderboard import render_leaderboard
from attention_lab.queue.ledger import QueueLedger
from attention_lab.queue.paths import default_db_path, default_pid_path, ensure_queue_dirs
from attention_lab.training.config import load_config


def _open_ledger(args: argparse.Namespace) -> QueueLedger:
    ensure_queue_dirs(args.root)
    db_path = Path(args.db) if args.db else default_db_path(args.root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = QueueLedger(db_path)
    ledger.initialize()
    return ledger


def _print_rows(rows: list[dict]) -> None:
    for row in rows:
        print(
            f"{row['id']}  {row['config_name']}  {row['attention_type']}  "
            f"{row['stage']}  {row['status']}  {row.get('failure_class') or ''}"
        )


def cmd_status(args: argparse.Namespace) -> None:
    ledger = _open_ledger(args)
    try:
        print(render_leaderboard(ledger.list_runs()), end="")
    finally:
        ledger.close()


def cmd_add(args: argparse.Namespace) -> None:
    paths = ensure_queue_dirs(args.root)
    ledger = _open_ledger(args)
    try:
        for source in args.config_paths:
            source_path = Path(source)
            load_config(source_path)
            dest = paths["inbox"] / source_path.name
            shutil.copy2(source_path, dest)
            result = ledger.scan_inbox(paths["inbox"])
            print(f"added: {dest} inserted={result['inserted']} skipped={result['skipped']}")
            for error in result["errors"]:
                print(f"error: {error['path']}: {error['error']}")
    finally:
        ledger.close()


def cmd_ls(args: argparse.Namespace) -> None:
    ledger = _open_ledger(args)
    try:
        _print_rows(ledger.list_runs(stage=args.stage, status=args.status))
    finally:
        ledger.close()


def cmd_show(args: argparse.Namespace) -> None:
    ledger = _open_ledger(args)
    try:
        row = ledger.get_run(args.run_id_or_name)
        if row is None:
            raise SystemExit(f"unknown run: {args.run_id_or_name}")
        for key, value in row.items():
            print(f"{key}: {value}")
        log_path = Path(row["run_dir"]) / "queue_runner.log"
        if log_path.exists():
            print("\nlast 20 queue log lines:")
            for line in log_path.read_text(encoding="utf-8").splitlines()[-20:]:
                print(line)
    finally:
        ledger.close()


def cmd_note(args: argparse.Namespace) -> None:
    ledger = _open_ledger(args)
    try:
        ledger.update_notes(args.run_id_or_name, args.text)
    finally:
        ledger.close()


def cmd_kill(args: argparse.Namespace) -> None:
    ledger = _open_ledger(args)
    try:
        row = ledger.get_run(args.run_id_or_name)
        if row is None:
            raise SystemExit(f"unknown run: {args.run_id_or_name}")
        if row["status"] == "RUNNING":
            pid_path = default_pid_path(args.root)
            if pid_path.exists():
                pid = int(pid_path.read_text(encoding="utf-8").strip())
                os.kill(pid, signal.SIGTERM)
        else:
            ledger.mark_failed(row["id"], failure_class="UNKNOWN", killed=True, notes="killed by operator")
    finally:
        ledger.close()


def cmd_requeue(args: argparse.Namespace) -> None:
    ledger = _open_ledger(args)
    try:
        ledger.requeue(args.run_id_or_name)
    finally:
        ledger.close()


def cmd_start(args: argparse.Namespace) -> None:
    subprocess.Popen(["bash", "scripts/queue_daemon.sh"], cwd=args.root)
    print("queue daemon start requested")


def cmd_stop(args: argparse.Namespace) -> None:
    pid_path = default_pid_path(args.root)
    if not pid_path.exists():
        print("queue daemon is not running")
        return
    pid = int(pid_path.read_text(encoding="utf-8").strip())
    os.kill(pid, signal.SIGTERM)
    print(f"sent SIGTERM to {pid}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="attn-queue")
    parser.add_argument("--root", default=".")
    parser.add_argument("--db", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status")
    status.set_defaults(func=cmd_status)

    add = subparsers.add_parser("add")
    add.add_argument("config_paths", nargs="+")
    add.set_defaults(func=cmd_add)

    ls = subparsers.add_parser("ls")
    ls.add_argument("--stage", choices=["SCREEN", "FULL"], default=None)
    ls.add_argument("--status", choices=["PENDING", "RUNNING", "PASSED", "FAILED", "KILLED"], default=None)
    ls.set_defaults(func=cmd_ls)

    show = subparsers.add_parser("show")
    show.add_argument("run_id_or_name")
    show.set_defaults(func=cmd_show)

    note = subparsers.add_parser("note")
    note.add_argument("run_id_or_name")
    note.add_argument("text")
    note.set_defaults(func=cmd_note)

    kill = subparsers.add_parser("kill")
    kill.add_argument("run_id_or_name")
    kill.set_defaults(func=cmd_kill)

    requeue = subparsers.add_parser("requeue")
    requeue.add_argument("run_id_or_name")
    requeue.set_defaults(func=cmd_requeue)

    start = subparsers.add_parser("start")
    start.set_defaults(func=cmd_start)

    stop = subparsers.add_parser("stop")
    stop.set_defaults(func=cmd_stop)

    leaderboard = subparsers.add_parser("leaderboard")
    leaderboard.add_argument("--min-stage", choices=["SCREEN", "FULL"], default=None)
    leaderboard.add_argument("--sort", choices=["loss", "ppl", "speed"], default=None)
    leaderboard.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
