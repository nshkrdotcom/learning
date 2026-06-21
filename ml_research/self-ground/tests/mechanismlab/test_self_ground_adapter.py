from __future__ import annotations

import csv
import json

from mechanismlab.core import ArtifactContract, ClaimSpec, ExperimentSpec
from mechanismlab.core.status import ClaimStatus
from self_ground.mechanismlab_adapter import (
    artifact_contract_for_phase3,
    build_mechanismlab_report_for_phase3,
    evidence_payload_from_phase3_run,
    phase3_experiment_spec,
    self_ground_claim_spec,
    write_mechanismlab_artifacts_for_phase3,
)


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path, rows) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_phase3_like_run(run_dir) -> None:
    run_dir.mkdir(exist_ok=True)
    _write_json(
        run_dir / "config.json",
        {
            "engine_backend": "transformer_lens",
            "sae_backend": "sae_lens",
            "evaluation_adapter": "negation_ravel_adapter",
        },
    )
    _write_json(run_dir / "compatibility.json", {"compatible": True, "diagnostic_only": False})
    _write_json(
        run_dir / "behavioral_task_validation.json",
        {"summary": {"valid_tasks": 6, "passes_minimum": True}},
    )
    _write_json(
        run_dir / "feature_sets.json",
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
                    "feature_ids": ["sae_3", "sae_4"],
                },
            ]
        },
    )
    _write_json(run_dir / "skipped_behavioral_rows.json", {"n_skipped_rows": 0})
    _write_json(
        run_dir / "mechanism_report.json",
        {
            "claim_status": "candidate_evidence",
            "limitations": [],
            "not_supported_claims": ["no broad claim"],
        },
    )
    _write_jsonl(run_dir / "behavioral_tasks.jsonl", [{"id": "task_1"}])
    _write_jsonl(run_dir / "excluded_behavioral_tasks.jsonl", [])
    _write_jsonl(run_dir / "baseline_task_scores.jsonl", [{"task_id": "task_1"}])
    _write_jsonl(
        run_dir / "behavioral_intervention_results.jsonl",
        [{"task_id": "task_1", "feature_set_label": "top"}],
    )
    _write_json(run_dir / "baseline_validation.json", {"finite": True})
    with (run_dir / "baseline_task_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["family", "n_tasks"])
        writer.writeheader()
        writer.writerow({"family": "__all__", "n_tasks": 6})
    with (run_dir / "behavioral_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "feature_set_label",
                "family",
                "target_absolute_delta_mean",
                "control_absolute_delta_mean",
                "norm_drift_warning_rate",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "feature_set_label": "top",
                "family": "__all__",
                "target_absolute_delta_mean": "0.3",
                "control_absolute_delta_mean": "0.05",
                "norm_drift_warning_rate": "0.0",
            }
        )
        writer.writerow(
            {
                "feature_set_label": "density_matched_seed_7",
                "family": "__all__",
                "target_absolute_delta_mean": "0.05",
                "control_absolute_delta_mean": "0.01",
                "norm_drift_warning_rate": "0.0",
            }
        )


def test_self_ground_adapter_specs_are_valid() -> None:
    assert isinstance(self_ground_claim_spec(), ClaimSpec)
    assert isinstance(phase3_experiment_spec(), ExperimentSpec)
    assert isinstance(artifact_contract_for_phase3(), ArtifactContract)
    assert self_ground_claim_spec().project == "self-ground"


def test_self_ground_adapter_builds_generic_report(tmp_path) -> None:
    _write_phase3_like_run(tmp_path)

    payload = evidence_payload_from_phase3_run(tmp_path)
    report = build_mechanismlab_report_for_phase3(tmp_path)

    assert payload["density_matched_control_count"] == 1
    assert report.status == ClaimStatus.CANDIDATE_EVIDENCE
    assert report.project == "self-ground"


def test_self_ground_adapter_writes_generic_artifacts(tmp_path) -> None:
    _write_phase3_like_run(tmp_path)

    report = write_mechanismlab_artifacts_for_phase3(tmp_path)

    assert report.status == ClaimStatus.CANDIDATE_EVIDENCE
    for name in [
        "mechanismlab_claim.json",
        "mechanismlab_experiment.json",
        "mechanismlab_run_manifest.json",
        "mechanismlab_claim_report.json",
        "mechanismlab_claim_report.md",
    ]:
        assert (tmp_path / name).exists(), name
