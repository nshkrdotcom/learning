from __future__ import annotations

import json

import pytest

from mechanismlab.core import RunManifest
from mechanismlab.trackers import LocalJsonTracker, OptionalDependencyUnavailable
from mechanismlab.trackers.mlflow_tracker import MLflowTracker
from mechanismlab.trackers.wandb_tracker import WandBTracker


def test_local_tracker_writes_jsonl_events(tmp_path) -> None:
    tracker = LocalJsonTracker(tmp_path)
    run = RunManifest(run_id="run.test")

    tracker.start_run(run)
    tracker.log_metrics({"effect": 1.0}, step=1)
    tracker.log_artifact(tmp_path / "tracker_events.jsonl")
    tracker.finish("ok")

    rows = [
        json.loads(line)
        for line in (tmp_path / "tracker_events.jsonl").read_text().splitlines()
    ]
    assert [row["event"] for row in rows] == [
        "start_run",
        "log_metrics",
        "log_artifact",
        "finish",
    ]


def test_wandb_tracker_missing_import_is_controlled(monkeypatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    with pytest.raises(OptionalDependencyUnavailable, match="wandb"):
        WandBTracker()
    assert WandBTracker.manifest().available is False


def test_mlflow_tracker_missing_import_is_controlled(monkeypatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    with pytest.raises(OptionalDependencyUnavailable, match="mlflow"):
        MLflowTracker()
    assert MLflowTracker.manifest().available is False
