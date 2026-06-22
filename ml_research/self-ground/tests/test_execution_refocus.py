from __future__ import annotations

import csv
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_decoded_sae_patch_nonzero import patch_check_notes
from scripts.check_run_capability import collect_run_capability
from scripts.diagnose_zero_effect_run import diagnose_zero_effect_run
from scripts.run_e002_negation_sae_density_matched import run_e002
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
                "control_suite": "matched_non_negation_current",
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
    (run_dir / "control_suite.json").write_text(
        json.dumps(
            {
                "control_suite": "matched_non_negation_current",
                "expanded_suites": ["matched_non_negation_current"],
                "n_control_cases": 1,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "control_task_mapping.jsonl").write_text(
        json.dumps(
            {
                "task_id": "task_1",
                "family": "sentiment_negation",
                "control_suite": "matched_non_negation_current",
                "control_case_id": "task_1_matched",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "control_suite_validation.json").write_text(
        json.dumps(
            {
                "requested_mode": "matched_non_negation_current",
                "expanded_suites": ["matched_non_negation_current"],
                "total_tasks": 1,
                "valid_control_cases": 1,
                "excluded_control_cases": 0,
                "passes_minimum": True,
            }
        ),
        encoding="utf-8",
    )
    with (run_dir / "selected_feature_rationale.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "feature_id", "score"])
        writer.writeheader()
        writer.writerow({"rank": 1, "feature_id": "sae_1", "score": 1.0})
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
                "control_suite",
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
                "control_suite": "matched_non_negation_current",
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
                "control_suite": "matched_non_negation_current",
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


def test_inspect_claim_run_classifies_serious_and_large_cpu_runs(tmp_path) -> None:
    serious = tmp_path / "serious"
    _write_synthetic_claim_run(serious)
    config = json.loads((serious / "config.json").read_text())
    config.update({"device": "cuda", "per_family": 10, "top_k_features": 5})
    (serious / "config.json").write_text(json.dumps(config), encoding="utf-8")
    feature_sets = json.loads((serious / "feature_sets.json").read_text())
    feature_sets["feature_sets"].extend(
        [
            {
                "label": "random_seed_7",
                "selection_method": "seeded_random_excluding_top_fraction",
                "feature_ids": ["sae_20", "sae_21"],
            },
            {
                "label": "bottom_active",
                "selection_method": "bottom_active_abs_score",
                "feature_ids": ["sae_30", "sae_31"],
            },
        ]
    )
    (serious / "feature_sets.json").write_text(json.dumps(feature_sets), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "scripts/inspect_claim_run.py", "--run-dir", str(serious), "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(result.stdout)["run_classification"] == "serious_gpu_evidence_run"

    large_cpu = tmp_path / "large_cpu"
    _write_synthetic_claim_run(large_cpu)
    config = json.loads((large_cpu / "config.json").read_text())
    config.update({"device": "cpu", "per_family": 3, "top_k_features": 5})
    (large_cpu / "config.json").write_text(json.dumps(config), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "scripts/inspect_claim_run.py", "--run-dir", str(large_cpu), "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(result.stdout)["run_classification"] == "large_cpu_diagnostic"


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


def test_capability_check_writes_cuda_blocker(tmp_path) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def device_count() -> int:
            return 0

    torch_module = ModuleType("torch")
    torch_module.__version__ = "test"
    torch_module.cuda = FakeCuda

    def importer(name: str) -> ModuleType:
        if name == "torch":
            return torch_module
        if name in {"transformer_lens", "sae_lens"}:
            return ModuleType(name)
        raise ModuleNotFoundError(name)

    result = collect_run_capability(
        out_dir=tmp_path,
        model="model",
        sae_release="release",
        sae_id="id",
        importer=importer,
    )

    assert result["can_attempt_e002_gpu"] is False
    assert any(blocker["type"] == "cuda_unavailable" for blocker in result["blockers"])
    assert (tmp_path / "capability.json").exists()


def test_e002_blocks_without_cuda_and_does_not_run_commands(tmp_path, monkeypatch) -> None:
    calls = []

    def capability(**kwargs):
        return {
            "can_attempt_e002_gpu": False,
            "blockers": [{"type": "cuda_unavailable"}],
            **kwargs,
        }

    def fail_run(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("subprocess should not run")

    monkeypatch.setattr(
        "scripts.run_e002_negation_sae_density_matched.collect_run_capability",
        capability,
    )
    monkeypatch.setattr("scripts.run_e002_negation_sae_density_matched.subprocess.run", fail_run)
    result = run_e002(
        Namespace(
            device="cuda",
            ranking_per_family=10,
            eval_per_family=10,
            ranking_top_k=50,
            eval_top_k=5,
            operations="ablate",
            random_seeds="7,11,13",
            out_root=str(tmp_path),
            allow_cpu_serious_run=False,
            force=False,
        )
    )

    assert result["status"] == "blocked"
    assert calls == []
    assert (Path(result["eval_dir"]) / "BLOCKED.json").exists()


def test_e002_cpu_fallback_requires_explicit_allowance(tmp_path, monkeypatch) -> None:
    commands = []

    def capability(**kwargs):
        return {"can_attempt_e002_gpu": False, "blockers": [], **kwargs}

    class Completed:
        stdout = json.dumps({"ok": True, "claim": {"claim_status": "insufficient_evidence"}})

    def record_run(command, **kwargs):
        commands.append(command)
        return Completed()

    monkeypatch.setattr(
        "scripts.run_e002_negation_sae_density_matched.collect_run_capability",
        capability,
    )
    monkeypatch.setattr("scripts.run_e002_negation_sae_density_matched.subprocess.run", record_run)
    result = run_e002(
        Namespace(
            device="cpu",
            ranking_per_family=3,
            eval_per_family=3,
            ranking_top_k=30,
            eval_top_k=5,
            operations="ablate",
            random_seeds="7,11,13",
            out_root=str(tmp_path),
            allow_cpu_serious_run=True,
            force=True,
        )
    )

    assert result["status"] == "completed"
    assert len(commands) == 3
    assert any(
        any("run_real_activation_ranking.py" in str(part) for part in command)
        for command in commands
    )
    assert any(
        any("run_negation_ravel_eval.py" in str(part) for part in command)
        for command in commands
    )


def _write_zero_effect_fixture(
    run_dir: Path,
    ranking_dir: Path,
    *,
    row_target: float,
    row_control: float,
    summary_target: float,
    summary_control: float,
    include_telemetry: bool = True,
) -> None:
    _write_synthetic_claim_run(run_dir)
    ranking_dir.mkdir()
    with (ranking_dir / "feature_rankings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "feature_id",
                "score",
                "abs_score",
                "mean_pos",
                "mean_neg",
                "mean_para",
                "mean_decoy",
            ],
        )
        writer.writeheader()
        for index in [1, 2, 10, 11]:
            writer.writerow(
                {
                    "feature_id": f"sae_{index}",
                    "score": "0.0",
                    "abs_score": "0.0",
                    "mean_pos": "0.0",
                    "mean_neg": "0.0",
                    "mean_para": "0.0",
                    "mean_decoy": "0.0",
                }
            )
    row = {
        "task_id": "task_1",
        "feature_set_label": "top",
        "target_absolute_delta": row_target,
        "control_absolute_delta": row_control,
    }
    if include_telemetry:
        row.update({"decoded_delta_norm_mean": 0.0, "relative_norm_drift_mean": 0.0})
    (run_dir / "behavioral_intervention_results.jsonl").write_text(
        json.dumps(row) + "\n",
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
                "target_absolute_delta_mean": summary_target,
                "control_absolute_delta_mean": summary_control,
                "specificity_gap_mean": summary_target - summary_control,
            }
        )


def test_zero_effect_diagnosis_detects_all_row_zeros(tmp_path) -> None:
    run_dir = tmp_path / "run"
    ranking_dir = tmp_path / "ranking"
    _write_zero_effect_fixture(
        run_dir,
        ranking_dir,
        row_target=0.0,
        row_control=0.0,
        summary_target=0.0,
        summary_control=0.0,
    )
    diagnosis = diagnose_zero_effect_run(
        run_dir=run_dir,
        ranking_dir=ranking_dir,
        out_dir=tmp_path / "out",
    )

    assert "all_row_deltas_zero" in diagnosis["diagnosis_labels"]
    assert "decoded_delta_norm_zero_or_missing" in diagnosis["diagnosis_labels"]


def test_zero_effect_diagnosis_detects_aggregation_hiding_nonzero_rows(tmp_path) -> None:
    run_dir = tmp_path / "run"
    ranking_dir = tmp_path / "ranking"
    _write_zero_effect_fixture(
        run_dir,
        ranking_dir,
        row_target=0.4,
        row_control=0.0,
        summary_target=0.0,
        summary_control=0.0,
    )
    diagnosis = diagnose_zero_effect_run(
        run_dir=run_dir,
        ranking_dir=ranking_dir,
        out_dir=tmp_path / "out",
    )

    assert "aggregation_hides_nonzero_rows" in diagnosis["diagnosis_labels"]


def test_patch_check_notes_identify_intervention_failure_modes() -> None:
    assert patch_check_notes(
        feature_delta_l1=0.0,
        decoded_delta_norm=0.0,
        max_abs_logit_delta=0.0,
    ) == ["selected_features_inactive_or_no_feature_change"]
    assert "possible_sae_decode_issue" in patch_check_notes(
        feature_delta_l1=1.0,
        decoded_delta_norm=0.0,
        max_abs_logit_delta=0.0,
    )
    assert "possible_patch_or_hook_issue" in patch_check_notes(
        feature_delta_l1=1.0,
        decoded_delta_norm=1.0,
        max_abs_logit_delta=0.0,
    )
    assert "decoded_patch_changes_logits" in patch_check_notes(
        feature_delta_l1=1.0,
        decoded_delta_norm=1.0,
        max_abs_logit_delta=0.1,
    )
