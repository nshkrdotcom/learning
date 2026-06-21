from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from self_ground.ravel_adapter.saebench_probe import probe_saebench_ravel_bridge


def test_framework_extraction_is_not_active() -> None:
    assert not Path("src/mechanismlab").exists()
    assert not Path("src/self_ground/mechanismlab_adapter.py").exists()
    assert not Path("scripts/write_mechanismlab_report.py").exists()
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "mechanismlab =" not in pyproject
    assert "src/mechanismlab" not in pyproject


def test_code_classification_records_framework_slop() -> None:
    text = Path("docs/code_classification.md").read_text(encoding="utf-8")
    assert "framework_slop" in text
    assert "`src/mechanismlab/*`" in text
    assert "`src/self_ground/real_behavioral_intervention.py`" in text
    assert "TransformerLens + SAELens" in text


def test_execution_stack_documents_existing_library_ownership() -> None:
    text = Path("docs/execution_stack.md").read_text(encoding="utf-8")
    assert "TransformerLens" in text
    assert "SAELens" in text
    assert '"engine_backend": "transformer_lens"' in text
    assert '"sae_backend": "sae_lens"' in text
    assert "generic activation patching framework" in text


def test_serious_experiment_spec_has_required_commands() -> None:
    text = Path("experiments/E002_real_negation_sae_density_matched_run.md").read_text(
        encoding="utf-8"
    )
    assert "EleutherAI/pythia-70m-deduped" in text
    assert "pythia-70m-deduped-res-sm" in text
    assert "--per-family 10" in text
    assert "--top-k-features 5" in text
    assert "--baseline-mode top-vs-random-density-and-bottom-active" in text
    assert "--device cuda" in text
    assert "diagnostic-only" in text


def _write_synthetic_claim_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "model_name": "EleutherAI/pythia-70m-deduped",
                "hook_point": "blocks.2.hook_resid_post",
                "sae_release": "pythia-70m-deduped-res-sm",
                "sae_id": "blocks.2.hook_resid_post",
                "engine_backend": "transformer_lens",
                "sae_backend": "sae_lens",
                "evaluation_adapter": "negation_ravel_adapter",
                "baseline_mode": "top-vs-density-matched-multiseed",
                "per_family": 2,
                "top_k_features": 2,
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "behavioral_tasks.jsonl").write_text(
        json.dumps({"id": "task_1", "family": "sentiment_negation"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "behavioral_task_validation.json").write_text(
        json.dumps(
            {
                "summary": {
                    "passes_minimum": True,
                    "total_tasks": 1,
                    "valid_tasks": 1,
                    "excluded_tasks": 0,
                    "valid_by_family": {"sentiment_negation": 1},
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "excluded_behavioral_tasks.jsonl").write_text("", encoding="utf-8")
    (run_dir / "compatibility.json").write_text(
        json.dumps(
            {
                "compatible": True,
                "metadata_compatible": True,
                "shape_compatible": True,
                "reconstruction_compatible": True,
                "diagnostic_only": False,
                "declared_model": "pythia-70m-deduped",
                "declared_hook_point": "blocks.2.hook_resid_post",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "feature_sets.json").write_text(
        json.dumps(
            {
                "feature_sets": [
                    {
                        "label": "top",
                        "selection_method": "ranking_abs_score_top_k",
                        "feature_ids": ["sae_1", "sae_2"],
                    },
                    {
                        "label": "density_matched_seed_7",
                        "selection_method": "activation_density_matched",
                        "feature_ids": ["sae_10", "sae_11"],
                        "matched_control_metadata": {
                            "stats_source": "per_condition_mean_approximation",
                            "relaxed": True,
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "baseline_task_scores.jsonl").write_text(
        json.dumps({"task_id": "task_1"}) + "\n",
        encoding="utf-8",
    )
    with (run_dir / "baseline_task_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["family", "n_tasks"])
        writer.writeheader()
        writer.writerow({"family": "sentiment_negation", "n_tasks": 1})
    (run_dir / "baseline_validation.json").write_text(
        json.dumps({"finite": True, "n_nonfinite_rows": 0}),
        encoding="utf-8",
    )
    (run_dir / "behavioral_intervention_results.jsonl").write_text(
        json.dumps({"task_id": "task_1", "feature_set_label": "top"}) + "\n",
        encoding="utf-8",
    )
    with (run_dir / "behavioral_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "feature_set_label",
                "operation",
                "factor",
                "patch_mode",
                "family",
                "target_absolute_delta_mean",
                "control_absolute_delta_mean",
                "specificity_gap_mean",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "feature_set_label": "top",
                "operation": "ablate",
                "factor": "",
                "patch_mode": "delta",
                "family": "__all__",
                "target_absolute_delta_mean": "0.3",
                "control_absolute_delta_mean": "0.1",
                "specificity_gap_mean": "0.2",
            }
        )
        writer.writerow(
            {
                "feature_set_label": "density_matched_seed_7",
                "operation": "ablate",
                "factor": "",
                "patch_mode": "delta",
                "family": "__all__",
                "target_absolute_delta_mean": "0.2",
                "control_absolute_delta_mean": "0.1",
                "specificity_gap_mean": "0.1",
            }
        )
    (run_dir / "skipped_behavioral_rows.json").write_text(
        json.dumps({"n_skipped_rows": 0, "reason_counts": {}, "examples": []}),
        encoding="utf-8",
    )
    (run_dir / "mechanism_report.json").write_text(
        json.dumps(
            {
                "claim_status": "insufficient_evidence",
                "recommended_claim": "No promotion beyond insufficient evidence.",
                "blocker_reason": None,
                "limitations": ["synthetic diagnostic fixture"],
            }
        ),
        encoding="utf-8",
    )


def test_inspect_claim_run_reports_artifact_summary(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_synthetic_claim_run(run_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_claim_run.py",
            "--run-dir",
            str(run_dir),
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["ok"] is True
    assert payload["run_classification"] == "diagnostic_or_smoke_run"
    assert payload["config"]["engine_backend"] == "transformer_lens"
    assert payload["config"]["sae_backend"] == "sae_lens"
    assert payload["feature_sets"]["density_matched_present"] is True
    assert payload["feature_sets"]["density_relaxed"] is True
    assert payload["claim"]["claim_status"] == "insufficient_evidence"
    assert payload["metrics"]["specificity_gap"] == 0.2


def test_inspect_claim_run_fails_on_missing_required_artifacts(tmp_path) -> None:
    run_dir = tmp_path / "missing"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/inspect_claim_run.py", "--run-dir", str(run_dir), "--json"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "compatibility.json" in payload["missing_required_artifacts"]


def test_saebench_probe_does_not_claim_integration_without_upstream_import(tmp_path) -> None:
    def missing_importer(name: str) -> ModuleType:
        raise ModuleNotFoundError(f"No module named {name!r}")

    result = probe_saebench_ravel_bridge(out_dir=tmp_path, importer=missing_importer)

    assert result.status == "not_installed"
    assert result.status != "bridge_feasible"
    assert result.status != "ran_real_eval"
