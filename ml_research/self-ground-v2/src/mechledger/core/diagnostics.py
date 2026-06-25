from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class DiagnosticSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    BLOCKING = "blocking"


class Diagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: DiagnosticSeverity
    code: str
    message: str
    file: str | None = None
    line: int | None = None
    object_id: str | None = None
    suggested_fix: str | None = None

    def format(self) -> str:
        prefix = self.severity.value.upper()
        location = self.file or "<unknown>"
        if self.line is not None:
            location += f":{self.line}"
        if self.object_id:
            location += f" {self.object_id}"
        text = f"{prefix} {location}\nRule: {self.code}\n{self.message}"
        if self.suggested_fix:
            text += f"\nSuggested fix: {self.suggested_fix}"
        return text


class DiagnosticError(ValueError):
    def __init__(self, diagnostic: Diagnostic) -> None:
        self.diagnostic = diagnostic
        super().__init__(diagnostic.format())


def raise_diagnostic(
    *,
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR,
    code: str,
    message: str,
    file: str | None = None,
    line: int | None = None,
    object_id: str | None = None,
    suggested_fix: str | None = None,
) -> None:
    raise DiagnosticError(
        Diagnostic(
            severity=severity,
            code=code,
            message=message,
            file=file,
            line=line,
            object_id=object_id,
            suggested_fix=suggested_fix,
        )
    )


def exit_code_for(diagnostics: list[Diagnostic]) -> int:
    if any(item.severity == DiagnosticSeverity.BLOCKING for item in diagnostics):
        return 1
    if any(item.severity == DiagnosticSeverity.ERROR for item in diagnostics):
        return 2
    return 0
