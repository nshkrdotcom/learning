from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mwb.domain.objects import PolicyEvaluationReport
from mwb.project import Project
from mwb.refs import stable_ref
from mwb.sqlite_index import initialize_schema, insert_payload

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class PolicyProfile:
    name: str
    zero_ablation_claim_ceiling: str
    require_resample_ablation_for_candidate: bool
    require_noising_and_denoising: bool
    require_static_space_type_checks: bool
    require_alternative_explanations: bool
    require_generalization_for_mechanism_word: bool
    unresolved_scientific_debt: str


BUILTIN_PROFILES = {
    "strict": PolicyProfile(
        name="strict",
        zero_ablation_claim_ceiling="diagnostic_only",
        require_resample_ablation_for_candidate=True,
        require_noising_and_denoising=True,
        require_static_space_type_checks=True,
        require_alternative_explanations=True,
        require_generalization_for_mechanism_word=True,
        unresolved_scientific_debt="block_stronger_claims",
    ),
    "exploratory": PolicyProfile(
        name="exploratory",
        zero_ablation_claim_ceiling="causal_necessity",
        require_resample_ablation_for_candidate=False,
        require_noising_and_denoising=False,
        require_static_space_type_checks=False,
        require_alternative_explanations=False,
        require_generalization_for_mechanism_word=False,
        unresolved_scientific_debt="caveat",
    ),
}


class PolicyProfileService:
    def __init__(self, project: Project | None = None) -> None:
        self.project = project

    def get_profile(self, name: str | None = None) -> PolicyProfile:
        profile_name = name or (self.load_project_profile().name if self.project else "strict")
        profile = BUILTIN_PROFILES.get(profile_name)
        if profile is None:
            expected = ", ".join(sorted(BUILTIN_PROFILES))
            raise ValueError(
                f"unknown policy profile {profile_name!r}; expected one of: {expected}"
            )
        return profile

    def load_project_profile(self) -> PolicyProfile:
        if self.project is None:
            return BUILTIN_PROFILES["strict"]
        project_toml = self.project.mechanism_dir / "project.toml"
        if not project_toml.exists():
            return BUILTIN_PROFILES["strict"]
        config = tomllib.loads(project_toml.read_text(encoding="utf-8"))
        profile_name = str(config.get("policy", {}).get("profile", "strict"))
        return self.get_profile(profile_name)

    def evaluate_verification(
        self,
        operations: list[JsonDict],
        *,
        claim_bearing: bool,
        profile_name: str | None = None,
    ) -> PolicyEvaluationReport:
        profile = self.get_profile(profile_name)
        operation_names = [str(operation.get("operation")) for operation in operations]
        checks = []
        blockers: list[str] = []
        warnings: list[str] = []
        claim_ceiling = None

        zero_check = {
            "name": "zero_ablation_claim_ceiling",
            "status": "pass",
            "operation_present": "zero_ablate" in operation_names,
            "ceiling": profile.zero_ablation_claim_ceiling,
        }
        if claim_bearing and "zero_ablate" in operation_names:
            if profile.zero_ablation_claim_ceiling == "diagnostic_only":
                zero_check["status"] = "fail"
                blockers.append("zero_ablation_claim_ceiling")
                claim_ceiling = "diagnostic_only"
            else:
                zero_check["status"] = "warn"
                warnings.append("zero_ablation_claim_ceiling")
                claim_ceiling = profile.zero_ablation_claim_ceiling
        checks.append(zero_check)

        resample_check = {
            "name": "resample_ablation_required",
            "status": "pass",
            "required": profile.require_resample_ablation_for_candidate,
            "operation_present": "resample_ablate" in operation_names,
        }
        if (
            claim_bearing
            and profile.require_resample_ablation_for_candidate
            and "resample_ablate" not in operation_names
        ):
            resample_check["status"] = "fail"
            blockers.append("missing_resample_ablation")
        checks.append(resample_check)

        paired_check = {
            "name": "noising_and_denoising_required",
            "status": "pass",
            "required": profile.require_noising_and_denoising,
            "has_noising": "noising" in operation_names,
            "has_denoising": "denoising" in operation_names,
        }
        if claim_bearing and profile.require_noising_and_denoising:
            if "noising" not in operation_names:
                blockers.append("missing_noising")
            if "denoising" not in operation_names:
                blockers.append("missing_denoising")
            if "missing_noising" in blockers or "missing_denoising" in blockers:
                paired_check["status"] = "fail"
        checks.append(paired_check)

        status = "fail" if blockers else "warn" if warnings else "pass"
        return PolicyEvaluationReport(
            wb_ref=stable_ref(
                "policy",
                profile.name,
                "verification",
                operation_names,
                claim_bearing,
                blockers,
                warnings,
            ),
            policy_profile=profile.name,
            status=status,
            mode="verification",
            blockers=_dedupe(blockers),
            warnings=_dedupe(warnings),
            claim_ceiling=claim_ceiling,
            checks=checks,
            profile=asdict(profile),
        )

    def evaluate_profile(self, profile_name: str | None = None) -> PolicyEvaluationReport:
        profile = self.get_profile(profile_name)
        checks = [
            {
                "name": "zero_ablation_claim_ceiling",
                "status": "pass",
                "value": profile.zero_ablation_claim_ceiling,
            },
            {
                "name": "noising_and_denoising_required",
                "status": "pass",
                "value": profile.require_noising_and_denoising,
            },
            {
                "name": "generalization_before_mechanism",
                "status": "pass",
                "value": profile.require_generalization_for_mechanism_word,
            },
        ]
        return PolicyEvaluationReport(
            wb_ref=stable_ref("policy", profile.name, "profile", checks),
            policy_profile=profile.name,
            status="pass",
            mode="profile",
            checks=checks,
            profile=asdict(profile),
        )

    def write_report(self, report: PolicyEvaluationReport) -> Path:
        if self.project is None:
            raise ValueError("PolicyProfileService.write_report requires a project")
        output_dir = self.project.mechanism_dir / "policy"
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = report.model_dump(mode="json")
        latest = output_dir / "latest_policy_evaluation.json"
        stable = output_dir / f"{report.wb_ref}.json"
        _write_json(latest, payload)
        _write_json(stable, payload)
        initialize_schema(self.project.sqlite_path)
        insert_payload(self.project.sqlite_path, "policy_evaluations", report.wb_ref, payload)
        return latest


def profile_for_name(name: str | None) -> PolicyProfile:
    return PolicyProfileService().get_profile(name or "strict")


def _write_json(path: Path, payload: JsonDict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _dedupe(values: list[str]) -> list[str]:
    deduped = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped
