from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mwb.domain.objects import (
    MechanisticUnitRef,
    SpaceCompatibilityReport,
    SpaceTransform,
    TensorSpace,
)
from mwb.project import Project
from mwb.refs import stable_ref
from mwb.sqlite_index import initialize_schema, insert_payload

PATCH_OPERATIONS = {"ablate", "resample_ablate", "amplify", "patch", "direct_patch"}


class MechanisticUnitRegistry:
    def __init__(self, units: list[MechanisticUnitRef]) -> None:
        self.units = {unit.wb_ref: unit for unit in units}

    def check_operation(self, unit_ref: str, operation: str) -> SpaceCompatibilityReport:
        unit = self.units[unit_ref]
        blockers: list[str] = []
        if operation in unit.invalid_operations:
            blockers.append("invalid_operation_for_unit")
        if unit.valid_operations and operation not in unit.valid_operations:
            blockers.append("invalid_operation_for_unit")
        return _report(
            operation=operation,
            status="fail" if blockers else "pass",
            unit_refs=[unit_ref],
            blockers=blockers,
        )


class SpaceTypeService:
    def __init__(self, project: Project | None = None) -> None:
        self.project = project

    def check_file(self, path: Path) -> SpaceCompatibilityReport:
        return self.check_payload(json.loads(path.read_text(encoding="utf-8")))

    def check_payload(self, payload: dict[str, Any]) -> SpaceCompatibilityReport:
        operation = str(payload["operation"])
        spaces = {
            item["wb_ref"]: TensorSpace.model_validate(item)
            for item in payload.get("spaces", [])
        }
        units = [
            MechanisticUnitRef.model_validate(item)
            for item in payload.get("units", [])
        ]
        transforms = [
            SpaceTransform.model_validate(item)
            for item in payload.get("transforms", [])
        ]
        blockers: list[str] = []
        transform_refs: list[str] = []
        required_transform: str | None = None

        registry = MechanisticUnitRegistry(units)
        for unit in units:
            unit_report = registry.check_operation(unit.wb_ref, operation)
            blockers.extend(unit_report.blockers)

        if operation == "compare_decoder_cosine":
            dictionary_refs = {
                str(unit.dictionary_ref)
                for unit in units
                if unit.unit_kind == "sae_feature" and unit.dictionary_ref
            }
            if len(dictionary_refs) > 1:
                blockers.append("incompatible_dictionary")

        source_ref = payload.get("source_space_ref")
        target_ref = payload.get("target_space_ref")
        source = spaces.get(str(source_ref)) if source_ref else None
        target = spaces.get(str(target_ref)) if target_ref else None
        if source_ref and source is None:
            blockers.append("unknown_source_space")
        if target_ref and target is None:
            blockers.append("unknown_target_space")
        if source is not None and target is not None:
            matching_transform = _matching_transform(transforms, source.wb_ref, target.wb_ref)
            if _normalization_context(source) != _normalization_context(target):
                if matching_transform is None:
                    blockers.append("normalization_context_mismatch")
                    required_transform = (
                        f"{_normalization_context(source)}_to_"
                        f"{_normalization_context(target)}"
                    )
                elif not matching_transform.provenance_ref:
                    blockers.append("missing_transform_provenance")
                else:
                    transform_refs.append(matching_transform.wb_ref)

        if operation in PATCH_OPERATIONS:
            for unit in units:
                unit_space_ref = unit.write_space_ref or unit.tensor_space_ref
                unit_space = spaces.get(unit_space_ref)
                if unit_space is None:
                    blockers.append("unknown_unit_space")
                    continue
                if target is None:
                    continue
                if unit_space.hook_point != target.hook_point:
                    blockers.append("wrong_hook_point")
                elif unit.hook_point and unit.hook_point != target.hook_point:
                    blockers.append("wrong_hook_point")

        deduped_blockers = _dedupe(blockers)
        return _report(
            operation=operation,
            status="fail" if deduped_blockers else "pass",
            source_space_ref=str(source_ref) if source_ref else None,
            target_space_ref=str(target_ref) if target_ref else None,
            unit_refs=[unit.wb_ref for unit in units],
            blockers=deduped_blockers,
            transform_refs=transform_refs,
            required_transform=required_transform,
        )

    def write_report(self, report: SpaceCompatibilityReport) -> Path:
        if self.project is None:
            raise ValueError("SpaceTypeService.write_report requires a project")
        output_dir = self.project.mechanism_dir / "space_checks"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "latest_space_check.json"
        payload = report.model_dump(mode="json")
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        initialize_schema(self.project.sqlite_path)
        insert_payload(self.project.sqlite_path, "space_checks", report.wb_ref, payload)
        return output


def _report(
    *,
    operation: str,
    status: str,
    source_space_ref: str | None = None,
    target_space_ref: str | None = None,
    unit_refs: list[str] | None = None,
    blockers: list[str] | None = None,
    transform_refs: list[str] | None = None,
    required_transform: str | None = None,
) -> SpaceCompatibilityReport:
    return SpaceCompatibilityReport(
        wb_ref=stable_ref(
            "spacecheck",
            operation,
            status,
            source_space_ref or "",
            target_space_ref or "",
            unit_refs or [],
            blockers or [],
            transform_refs or [],
            required_transform or "",
        ),
        operation=operation,
        status=status,
        source_space_ref=source_space_ref,
        target_space_ref=target_space_ref,
        unit_refs=list(unit_refs or []),
        blockers=list(blockers or []),
        transform_refs=list(transform_refs or []),
        required_transform=required_transform,
    )


def _matching_transform(
    transforms: list[SpaceTransform],
    source_ref: str,
    target_ref: str,
) -> SpaceTransform | None:
    for transform in transforms:
        if transform.from_space_ref == source_ref and transform.to_space_ref == target_ref:
            return transform
    return None


def _normalization_context(space: TensorSpace) -> str | None:
    return space.normalization_context or space.layernorm_context


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped
