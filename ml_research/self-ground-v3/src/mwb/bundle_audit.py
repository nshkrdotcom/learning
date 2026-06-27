from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from mwb.domain.objects import (
    BundleRebalanceProposal,
    ControlContaminationReport,
    ExampleGeometryReport,
)
from mwb.project import Project
from mwb.refs import stable_ref
from mwb.sqlite_index import initialize_schema, insert_payload

JsonDict = dict[str, Any]
NEGATION_MARKERS = (" not ", " not.", "n't", "never", "false that")
DEFAULT_BASELINE_MARGIN_MIN = 0.1
SELF_GROUND_E004_FORENSICS = Path(
    "/home/home/p/g/n/learning/ml_research/self-ground/"
    "runs/e004_specificity_rescue_matrix/forensics/forensics_summary.md"
)


class BundleAuditService:
    def __init__(self, project: Project | None = None) -> None:
        self.project = project

    def audit_bundle(self, name: str) -> ExampleGeometryReport:
        return self.audit_payload(_load_bundle_payload(name))

    def audit_payload(self, payload: dict[str, Any]) -> ExampleGeometryReport:
        bundle_name = str(payload["name"])
        examples = list(payload.get("examples", []))
        controls = dict(payload.get("controls", {}))
        checks: list[dict[str, Any]] = []
        blockers: list[str] = []
        warnings: list[str] = []
        proposals: list[dict[str, Any]] = []

        token_check = _token_validity_check(examples, controls)
        checks.append(token_check)
        if token_check["status"] == "fail":
            blockers.append("token_validation_failed")

        role_check = _role_balance_check(examples, controls)
        checks.append(role_check)
        if role_check["status"] == "warn":
            warnings.append("role_balance_low")
            proposals.extend(_control_proposals(role_check))

        contamination = _contamination_report(bundle_name, controls)
        checks.append(
            {
                "name": "control_contamination",
                "status": contamination.status,
                "contaminated_count": contamination.contaminated_count,
            }
        )
        if contamination.status == "fail":
            blockers.append("control_contamination")

        margin_check = _baseline_margin_check(examples, controls)
        checks.append(margin_check)
        if margin_check["status"] == "fail":
            blockers.append("baseline_margin_low")
        elif margin_check["status"] == "warn":
            warnings.append("baseline_margin_missing")

        proposals.extend(_heldout_proposals(payload))
        blockers = _dedupe(blockers)
        warnings = _dedupe(warnings)
        status = "fail" if blockers else "warn" if warnings else "pass"
        report = ExampleGeometryReport(
            wb_ref=stable_ref(
                "geometry",
                bundle_name,
                status,
                json.dumps(checks, sort_keys=True),
                json.dumps(proposals, sort_keys=True),
            ),
            bundle_name=bundle_name,
            status=status,
            checks=checks,
            blockers=blockers,
            warnings=warnings,
            proposals=proposals,
            contamination_report=contamination.model_dump(mode="json"),
            source_links=_source_links(),
        )
        return report

    def write_report(self, report: ExampleGeometryReport) -> Path:
        if self.project is None:
            raise ValueError("BundleAuditService.write_report requires a project")
        output_dir = self.project.mechanism_dir / "bundle_audits"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "latest_bundle_audit.json"
        payload = report.model_dump(mode="json")
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        initialize_schema(self.project.sqlite_path)
        insert_payload(
            self.project.sqlite_path,
            "example_geometry_reports",
            report.wb_ref,
            payload,
        )
        contamination = report.contamination_report
        if contamination.get("wb_ref"):
            insert_payload(
                self.project.sqlite_path,
                "control_contamination_reports",
                str(contamination["wb_ref"]),
                contamination,
            )
        return output

    def rebalance_bundle(self, name: str, *, dry_run: bool) -> BundleRebalanceProposal:
        return self.rebalance_payload(_load_bundle_payload(name), dry_run=dry_run)

    def rebalance_payload(
        self,
        payload: dict[str, Any],
        *,
        dry_run: bool,
    ) -> BundleRebalanceProposal:
        report = self.audit_payload(payload)
        proposal = BundleRebalanceProposal(
            wb_ref=stable_ref(
                "rebalance",
                payload["name"],
                dry_run,
                json.dumps(report.proposals, sort_keys=True),
            ),
            bundle_name=str(payload["name"]),
            dry_run=dry_run,
            proposals=report.proposals,
            source_report_ref=report.wb_ref,
        )
        return proposal

    def write_rebalance(self, proposal: BundleRebalanceProposal) -> Path:
        if self.project is None:
            raise ValueError("BundleAuditService.write_rebalance requires a project")
        output_dir = self.project.mechanism_dir / "bundle_rebalance"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "latest_rebalance_proposal.json"
        payload = proposal.model_dump(mode="json")
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        initialize_schema(self.project.sqlite_path)
        insert_payload(
            self.project.sqlite_path,
            "bundle_rebalance_proposals",
            proposal.wb_ref,
            payload,
        )
        return output


def _load_bundle_payload(name: str) -> dict[str, Any]:
    normalized = name.removeprefix("negation_")
    if normalized != "phase3_calibrated":
        raise ValueError(f"unknown built-in bundle: {name}")
    bundle_file = resources.files("mwb.resources.bundles").joinpath(
        "negation_phase3_calibrated.yaml"
    )
    return YAML(typ="safe").load(bundle_file.read_text(encoding="utf-8"))


def _token_validity_check(examples: list[JsonDict], controls: dict[str, list[JsonDict]]) -> dict:
    invalid: list[dict[str, str]] = []
    for row in [*examples, *[item for rows in controls.values() for item in rows]]:
        for field in ("id", "prompt", "target"):
            if not str(row.get(field, "")).strip():
                invalid.append({"id": str(row.get("id", "")), "field": field})
    return {
        "name": "token_validity",
        "status": "fail" if invalid else "pass",
        "invalid_count": len(invalid),
        "invalid": invalid,
    }


def _role_balance_check(examples: list[JsonDict], controls: dict[str, list[JsonDict]]) -> dict:
    target_count = len(examples)
    family_counts = {family: len(rows) for family, rows in controls.items()}
    low_families = [
        family
        for family, count in family_counts.items()
        if count < target_count
    ]
    return {
        "name": "role_balance",
        "status": "warn" if low_families else "pass",
        "target_count": target_count,
        "control_family_counts": family_counts,
        "low_families": low_families,
    }


def _contamination_report(
    bundle_name: str,
    controls: dict[str, list[JsonDict]],
) -> ControlContaminationReport:
    rows: list[dict[str, str]] = []
    for row in controls.get("negation_removed", []):
        prompt = f" {str(row.get('prompt', '')).lower()} "
        if any(marker in prompt for marker in NEGATION_MARKERS):
            rows.append(
                {
                    "id": str(row.get("id", "")),
                    "family": "negation_removed",
                    "reason": "negation_marker_in_negation_removed_control",
                }
            )
    status = "fail" if rows else "pass"
    return ControlContaminationReport(
        wb_ref=stable_ref("contam", bundle_name, rows),
        bundle_name=bundle_name,
        status=status,
        contaminated_count=len(rows),
        rows=rows,
    )


def _baseline_margin_check(examples: list[JsonDict], controls: dict[str, list[JsonDict]]) -> dict:
    low_rows: list[dict[str, Any]] = []
    missing = 0
    for row in [*examples, *[item for rows in controls.values() for item in rows]]:
        if "baseline_margin" not in row:
            missing += 1
            continue
        margin = float(row["baseline_margin"])
        if margin < DEFAULT_BASELINE_MARGIN_MIN:
            low_rows.append({"id": row.get("id"), "baseline_margin": margin})
    if low_rows:
        status = "fail"
    elif missing:
        status = "warn"
    else:
        status = "pass"
    return {
        "name": "baseline_margin",
        "status": status,
        "min_required": DEFAULT_BASELINE_MARGIN_MIN,
        "low_count": len(low_rows),
        "missing_count": missing,
        "low_rows": low_rows,
    }


def _control_proposals(role_check: dict[str, Any]) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    target_count = int(role_check["target_count"])
    for family in role_check["low_families"]:
        observed = int(role_check["control_family_counts"][family])
        proposals.append(
            {
                "kind": "add_control_examples",
                "family": family,
                "observed": observed,
                "target": target_count,
                "needed": max(target_count - observed, 0),
            }
        )
    return proposals


def _heldout_proposals(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "kind": "heldout_template",
            "domain": payload.get("domain", "unknown"),
            "rationale": "hold out prompt templates before mechanism-level claims",
        },
        {
            "kind": "heldout_vocabulary",
            "domain": payload.get("domain", "unknown"),
            "rationale": "hold out target/foil vocabulary before mechanism-level claims",
        },
    ]


def _source_links() -> dict[str, str]:
    links: dict[str, str] = {}
    if SELF_GROUND_E004_FORENSICS.exists():
        links["self_ground_e004_forensics"] = str(SELF_GROUND_E004_FORENSICS)
    return links


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped
