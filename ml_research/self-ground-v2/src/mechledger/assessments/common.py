from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mechledger.core.debt import DebtSeverity


class ThresholdSource(StrEnum):
    TOOL_DEFAULT = "tool_default"
    PROJECT_DEFAULT = "project_default"
    DECISION_JUSTIFIED = "decision_justified"


class AssessmentConditionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition_id: str
    condition_type: str
    passed: bool
    parameters: dict[str, Any] = Field(default_factory=dict)
    threshold_source: ThresholdSource | None = ThresholdSource.TOOL_DEFAULT
    threshold_justification: str | None = None
    threshold_decision_id: str | None = None
    failure_message: str
    default_consequence: Literal["blocker", "scientific_debt", "warning"]
    debt_type: str | None = None
    severity: DebtSeverity | None = None


class AssessmentReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessment_id: str
    conditions: dict[str, AssessmentConditionResult]

    @property
    def passed(self) -> bool:
        return all(condition.passed for condition in self.conditions.values())
