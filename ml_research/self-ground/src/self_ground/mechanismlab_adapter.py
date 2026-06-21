from __future__ import annotations

import csv
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from mechanismlab.core import (
    ArtifactContract,
    ClaimSpec,
    ExperimentSpec,
    RunManifest,
    write_model,
)
from mechanismlab.reports import ClaimReport, build_claim_report, write_claim_report_markdown
from self_ground.mechanism_report import REQUIRED_EVIDENCE_ARTIFACTS


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _git_info() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = None
    return {"commit": commit}


def self_ground_claim_spec() -> ClaimSpec:
    return ClaimSpec(
        claim_id="self_ground.negation_scope.sae_feature_set.v1",
        claim_type="feature_level_causal_effect",
        title="SELF-GROUND negation-scope SAE feature-set claim",
        claim_text=(
            "A selected SAE feature set has a token-contrast effect on negation-scope "
            "tasks under decoded residual intervention, subject to artifact-backed controls."
        ),
        project="self-ground",
        concept={"name": "negation_scope"},
        internal_object={"kind": "sae_feature_set"},
        intervention={"kind": "decoded_sae_residual_patch"},
        expected_effect={"kind": "target_greater_than_matched_controls"},
        controls_required=[
            "matched_non_negation_controls",
            "activation_density_matched_feature_sets",
        ],
        promotion_rules={
            "candidate": "requires complete artifacts, compatibility, effects, and controls",
            "strong": "requires project-specific SELF-GROUND mechanism report gates",
        },
        falsification_conditions=[
            "decoded intervention has no measurable target effect under matched controls",
            "control movement dominates target movement",
        ],
        metadata={"adapter": "self_ground.mechanismlab_adapter"},
    )


def phase3_experiment_spec() -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="self_ground.phase3.token_contrast.v1",
        claim_id="self_ground.negation_scope.sae_feature_set.v1",
        hypothesis=(
            "Selected SAE features move negation-scope token contrasts more than matched "
            "non-negation controls and deterministic feature-set controls."
        ),
        backend_requirements={
            "model_backend": "transformer_lens",
            "representation_backend": "sae_lens",
            "evaluation_adapter": "negation_ravel_adapter",
        },
        task_suite={"project": "self-ground", "concept": "negation_scope"},
        object_selection={"kind": "sae_feature_ranking"},
        interventions=[{"kind": "decoded_sae_residual_patch"}],
        evaluators=[{"kind": "token_contrast"}, {"kind": "claim_ledger"}],
        required_controls=[
            "matched_non_negation_controls",
            "activation_density_matched_feature_sets",
        ],
        required_artifacts=list(REQUIRED_EVIDENCE_ARTIFACTS),
        success_criteria={"claim_status": ["candidate_evidence", "strong_candidate_evidence"]},
        failure_criteria=["missing_artifacts", "compatibility_failed", "nonfinite_baseline"],
        metadata={"source": "self_ground.phase3"},
    )


def artifact_contract_for_phase3() -> ArtifactContract:
    return ArtifactContract(required=list(REQUIRED_EVIDENCE_ARTIFACTS))


def evidence_payload_from_phase3_run(run_dir: Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    compatibility = _read_json(run_path / "compatibility.json")
    validation = _read_json(run_path / "behavioral_task_validation.json")
    feature_sets = _read_json(run_path / "feature_sets.json")
    skipped = _read_json(run_path / "skipped_behavioral_rows.json")
    self_ground_report = _read_json(run_path / "mechanism_report.json")
    summary_rows = [
        row
        for row in _read_csv(run_path / "behavioral_summary.csv")
        if row.get("family") == "__all__"
    ]
    top_rows = [row for row in summary_rows if row.get("feature_set_label") == "top"]
    top_row = top_rows[0] if top_rows else {}
    density_count = len(
        [
            row
            for row in summary_rows
            if str(row.get("feature_set_label", "")).startswith("density_matched_seed_")
        ]
    )
    non_top_count = len(
        {
            row.get("feature_set_label")
            for row in summary_rows
            if row.get("feature_set_label") != "top"
        }
    )
    target_effect = float(top_row.get("target_absolute_delta_mean") or 0.0)
    control_effect = float(top_row.get("control_absolute_delta_mean") or 0.0)
    warning_rate = float(top_row.get("norm_drift_warning_rate") or 0.0)
    validation_summary = validation.get("summary", validation)
    n_tasks = int(validation_summary.get("valid_tasks") or top_row.get("n_tasks") or 0)
    controls_passed = density_count > 0 and target_effect > control_effect
    return {
        "compatibility": {
            "compatible": bool(compatibility.get("compatible")),
            "diagnostic_only": bool(compatibility.get("diagnostic_only")),
        },
        "n_tasks": n_tasks,
        "n_controls": density_count if density_count else non_top_count,
        "density_matched_control_count": density_count,
        "effect_abs": target_effect,
        "target_effect": target_effect,
        "control_effect": control_effect,
        "controls_passed": controls_passed,
        "warning_rate": warning_rate,
        "skipped_row_count": int(skipped.get("n_skipped_rows", 0) or 0),
        "self_ground_claim_status": self_ground_report.get("claim_status"),
        "limitations": list(self_ground_report.get("limitations", [])),
        "unsupported_claims": list(self_ground_report.get("not_supported_claims", [])),
        "project_specific": {
            "feature_sets": feature_sets.get("feature_sets", []),
            "summary_rows": summary_rows,
        },
    }


def phase3_run_manifest(run_dir: Path) -> RunManifest:
    run_path = Path(run_dir)
    config = _read_json(run_path / "config.json")
    artifacts = {
        path.name: str(path)
        for path in sorted(run_path.iterdir())
        if path.is_file()
    }
    return RunManifest(
        run_id=run_path.name,
        claim_id="self_ground.negation_scope.sae_feature_set.v1",
        experiment_id="self_ground.phase3.token_contrast.v1",
        project="self-ground",
        git=_git_info(),
        environment={"python": platform.python_version()},
        backends={
            "model": config.get("engine_backend"),
            "representation": config.get("sae_backend"),
            "evaluation": config.get("evaluation_adapter"),
        },
        artifacts=artifacts,
        status="completed" if (run_path / "mechanism_report.json").exists() else None,
        metadata={"source_run_dir": str(run_path)},
    )


def build_mechanismlab_report_for_phase3(run_dir: Path) -> ClaimReport:
    return build_claim_report(
        run_dir=Path(run_dir),
        claim=self_ground_claim_spec(),
        experiment=phase3_experiment_spec(),
        artifact_contract=artifact_contract_for_phase3(),
        evidence_payload=evidence_payload_from_phase3_run(Path(run_dir)),
    )


def write_mechanismlab_artifacts_for_phase3(run_dir: Path) -> ClaimReport:
    run_path = Path(run_dir)
    claim = self_ground_claim_spec()
    experiment = phase3_experiment_spec()
    write_model(claim, run_path / "mechanismlab_claim.json")
    write_model(experiment, run_path / "mechanismlab_experiment.json")
    report = build_mechanismlab_report_for_phase3(run_path)
    write_model(report, run_path / "mechanismlab_claim_report.json")
    write_claim_report_markdown(report, run_path / "mechanismlab_claim_report.md")
    manifest = phase3_run_manifest(run_path)
    write_model(manifest, run_path / "mechanismlab_run_manifest.json")
    return report
