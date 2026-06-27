from __future__ import annotations

import json
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from mwb.hashing import sha256_text
from mwb.time import utc_now

JsonDict = dict[str, Any]
EvidenceRelation = Literal[
    "supports",
    "contradicts",
    "depends_on",
    "derived_from",
    "tested_by",
    "confounded_by",
    "fails_on",
    "generalizes_to",
    "cited_by",
]
HypothesisWorkflowState = Literal[
    "noticed",
    "triaged",
    "structurally_plausible",
    "cheap_proxy_supported",
    "exact_patch_supported",
    "control_clean",
    "generalized",
    "claimable",
    "structurally_impossible",
    "proxy_false_positive",
    "control_leaky",
    "self_repair_confounded",
    "off_manifold",
    "task_artifact",
    "dictionary_artifact",
    "abandoned",
]


@runtime_checkable
class WorkbenchObjectProtocol(Protocol):
    wb_ref: str
    wb_type: str
    wb_version: str

    def wb_fingerprint(self) -> str: ...

    def wb_summary(self) -> JsonDict: ...

    def wb_parents(self) -> list[str]: ...

    def wb_artifacts(self) -> list[str]: ...


class WorkbenchObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wb_ref: str
    wb_type: str
    wb_version: str = "1"
    created_at: str = Field(default_factory=utc_now)
    parents: list[str] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)

    def wb_summary(self) -> JsonDict:
        return self.model_dump(mode="json")

    def wb_fingerprint(self) -> str:
        summary = self.wb_summary()
        encoded = json.dumps(summary, sort_keys=True, separators=(",", ":"))
        return sha256_text(encoded)

    def wb_parents(self) -> list[str]:
        return list(self.parents)

    def wb_artifacts(self) -> list[str]:
        refs = self.metadata.get("artifact_refs", [])
        if not isinstance(refs, list):
            return []
        return [str(ref) for ref in refs]


class Session(WorkbenchObject):
    wb_type: str = "Session"
    surface: str
    mode: str = "scratch"
    started_at: str = Field(default_factory=utc_now)
    ended_at: str | None = None


class CellEvent(WorkbenchObject):
    wb_type: str = "CellEvent"
    session_ref: str
    execution_index: int
    source_hash: str
    status: str
    created_object_refs: list[str] = Field(default_factory=list)
    mutated_object_refs: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)


class ModelIdentity(WorkbenchObject):
    wb_type: str = "ModelIdentity"
    _backend_model: Any = PrivateAttr(default=None)

    provider: str
    model_name: str
    backend: str
    backend_version: str | None = None
    checkpoint_hash: str | None = None
    revision: str | None = None
    tokenizer_ref: str | None = None
    config_hash: str | None = None


class DictionaryIdentity(WorkbenchObject):
    wb_type: str = "DictionaryIdentity"
    _backend_sae: Any = PrivateAttr(default=None)

    provider: str
    sae_id: str
    release: str
    hook_point: str
    model_ref: str
    feature_count: int | None = None
    dictionary_hash: str | None = None


class TensorSpace(WorkbenchObject):
    wb_type: str = "TensorSpace"
    model_ref: str
    hook_point: str
    axis_names: list[str]
    dtype: str
    shape: list[int | None]
    layernorm_context: str | None = None


class MechanisticUnitRef(WorkbenchObject):
    wb_type: str = "MechanisticUnitRef"
    unit_kind: str
    model_ref: str
    tensor_space_ref: str
    dictionary_ref: str | None = None
    feature_index: int | None = None
    hook_point: str | None = None


class ExampleBundle(WorkbenchObject):
    wb_type: str = "ExampleBundle"
    name: str
    domain: str
    examples: list[JsonDict]
    source: str
    bundle_hash: str | None = None


class ControlBundle(WorkbenchObject):
    wb_type: str = "ControlBundle"
    name: str
    domain: str
    control_families: dict[str, list[JsonDict]]
    source: str
    bundle_hash: str | None = None


class DomainBundle(WorkbenchObject):
    wb_type: str = "DomainBundle"
    name: str
    domain: str
    targets: ExampleBundle
    controls: ControlBundle
    source: str
    bundle_hash: str


class ActivationSet(WorkbenchObject):
    wb_type: str = "ActivationSet"
    _activation: Any = PrivateAttr(default=None)

    model_ref: str
    bundle_ref: str
    hook_point: str
    tensor_space_ref: str
    activation_summary: JsonDict


class FeatureRanking(WorkbenchObject):
    wb_type: str = "FeatureRanking"
    dictionary_ref: str
    activation_ref: str
    contrast: str
    rows: list[JsonDict]


class Hypothesis(WorkbenchObject):
    wb_type: str = "Hypothesis"
    title: str
    units: list[str]
    example_bundle_ref: str
    control_bundle_ref: str
    expected_effect: str
    required_controls: list[str]
    alternative_explanations: list[str] = Field(default_factory=list)
    requested_evidence_tier: str = "causal_necessity"


class HypothesisState(WorkbenchObject):
    wb_type: str = "HypothesisState"
    hypothesis_ref: str
    state: HypothesisWorkflowState
    evidence_tier: str = "none"
    claim_status: str | None = None
    approved_by: str | None = None
    decision_ref: str | None = None


class HypothesisTransitionReceipt(WorkbenchObject):
    wb_type: str = "HypothesisTransitionReceipt"
    hypothesis_ref: str
    from_state: HypothesisWorkflowState
    to_state: HypothesisWorkflowState
    evidence_tier: str = "none"
    claim_status: str | None = None
    approved_by: str | None = None
    decision_ref: str | None = None
    reason: str | None = None


class AlternativeExplanation(WorkbenchObject):
    wb_type: str = "AlternativeExplanation"
    hypothesis_ref: str
    explanation_id: str
    source_ref: str
    blocker: str
    evidence_for: list[str] = Field(default_factory=list)
    evidence_against: list[str] = Field(default_factory=list)
    next_test: str
    status: str = "live"


class PredictionLock(WorkbenchObject):
    wb_type: str = "PredictionLock"
    hypothesis_ref: str
    hypothesis_spec_hash: str
    expected_direction: str
    expected_controls: JsonDict
    git_state: JsonDict
    environment: JsonDict
    operator_ref: str = "local_user"


class InterventionSpec(WorkbenchObject):
    wb_type: str = "InterventionSpec"
    hypothesis_ref: str
    operation: str
    patch_mode: str
    target_units: list[str]
    metrics: list[str]


class PreflightReport(WorkbenchObject):
    wb_type: str = "PreflightReport"
    hypothesis_ref: str
    status: str
    checks: list[JsonDict]
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class VerificationRun(WorkbenchObject):
    wb_type: str = "VerificationRun"
    hypothesis_ref: str
    prediction_lock_ref: str | None
    status: str
    evidence_posture: str
    metrics: JsonDict = Field(default_factory=dict)
    artifact_refs: list[str] = Field(default_factory=list)


class BlockerReport(WorkbenchObject):
    wb_type: str = "BlockerReport"
    run_ref: str
    blockers: list[str]
    primary_blocker: str | None = None
    blocking_metrics: list[JsonDict] = Field(default_factory=list)


class NextProbePlan(WorkbenchObject):
    wb_type: str = "NextProbePlan"
    source_run_ref: str
    status: str
    diagnosis: JsonDict
    recommendation: JsonDict
    claim_implication: JsonDict = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)


class MechanismCard(WorkbenchObject):
    wb_type: str = "MechanismCard"
    title: str
    status: str
    evidence_tier: str
    run_ref: str | None = None
    allowed_language: list[str] = Field(default_factory=list)
    blocked_language: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)


class Claim(WorkbenchObject):
    wb_type: str = "Claim"
    text: str
    mechanism_card_ref: str
    evidence_tier: str
    status: str


class EvidenceEdge(WorkbenchObject):
    wb_type: str = "EvidenceEdge"
    src_ref: str
    dst_ref: str
    relation: EvidenceRelation
    source_ref: str | None = None
    source_path: str | None = None


_TYPE_REGISTRY: dict[str, type[WorkbenchObject]] = {
    cls.model_fields["wb_type"].default: cls
    for cls in [
        WorkbenchObject,
        Session,
        CellEvent,
        ModelIdentity,
        DictionaryIdentity,
        TensorSpace,
        MechanisticUnitRef,
        ExampleBundle,
        ControlBundle,
        DomainBundle,
        ActivationSet,
        FeatureRanking,
        Hypothesis,
        HypothesisState,
        HypothesisTransitionReceipt,
        AlternativeExplanation,
        PredictionLock,
        InterventionSpec,
        PreflightReport,
        VerificationRun,
        BlockerReport,
        NextProbePlan,
        MechanismCard,
        Claim,
        EvidenceEdge,
    ]
}


def is_workbench_object(value: object) -> bool:
    return isinstance(value, WorkbenchObjectProtocol)


def object_from_dict(payload: JsonDict) -> WorkbenchObject:
    wb_type = payload.get("wb_type")
    cls = _TYPE_REGISTRY.get(str(wb_type))
    if cls is None:
        raise ValueError(f"unknown workbench object type: {wb_type}")
    return cls.model_validate(payload)
