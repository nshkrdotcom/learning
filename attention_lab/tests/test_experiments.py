from __future__ import annotations

from pathlib import Path

from attention_lab.training.experiments import EXPERIMENT_STATUSES, get_experiment, list_experiments, main


def test_experiment_manifest_loads_e001():
    experiments = list_experiments()
    ids = [experiment["id"] for experiment in experiments]
    assert "E001_cp_trilinear_attention" in ids


def test_e001_paths_are_committed_and_status_is_valid():
    experiment = get_experiment("E001_cp_trilinear_attention")
    assert experiment["status"] in EXPERIMENT_STATUSES
    for key in ("plan", "config_dir", "report_dir", "baseline_config", "accurate_baseline_alias", "dataset_manifest"):
        assert Path(experiment[key]).exists(), key


def test_list_experiments_prints_e001(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["list_experiments.py", "--id", "E001_cp_trilinear_attention"])
    main()
    output = capsys.readouterr().out
    assert "experiment id: E001_cp_trilinear_attention" in output
    assert "dataset manifest: data/fineweb_edu_100m/manifest.json" in output
