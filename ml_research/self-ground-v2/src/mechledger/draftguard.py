from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mechledger.core.claim_ledger import ClaimLedger, parse_claim_ledger
from mechledger.core.draft_severity import DraftSeverity

TAG_PATTERN = re.compile(
    r"(?P<markdown>\[CLAIM:(?P<md_id>C[0-9]+[A-Za-z0-9_-]*)\])"
    r"|(?P<latex>\\claim\{(?P<tex_id>C[0-9]+[A-Za-z0-9_-]*)\})"
    r"|(?P<html><!--\s*CLAIM:(?P<html_id>C[0-9]+[A-Za-z0-9_-]*)\s*-->)"
)
MALFORMED_TAG_PATTERN = re.compile(
    r"\[CLAIM:(?!C[0-9]+[A-Za-z0-9_-]*\])([^\]]*)\]"
    r"|\\claim\{(?!C[0-9]+[A-Za-z0-9_-]*\})([^}]*)\}"
    r"|<!--\s*CLAIM:(?!C[0-9]+[A-Za-z0-9_-]*\s*-->)(.*?)-->"
)
SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)", re.MULTILINE)
MARKDOWN_OVERRIDE = re.compile(
    r"<!--\s*mechledger-disable\s+(?P<violation>[a-z_]+)\s*:\s*(?P<reason>.*?)\s*-->"
)
LATEX_OVERRIDE = re.compile(
    r"%\s*mechledger-disable\s+(?P<violation>[a-z_]+)\s*:\s*(?P<reason>.*)$",
    re.MULTILINE,
)
NEGATED_CLAIM_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bdo\s+not\s+claim\b",
        r"\bdo\s+not\s+assert\b",
        r"\bnot\s+claiming\b",
        r"\bno\s+claim\s+of\b",
    ]
]


class DraftTag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    raw: str
    start: int
    end: int
    line: int
    tag_type: str


class DraftOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str | None = None
    line: int | None
    claim_id: str
    violation_type: str
    reason: str
    raw_comment: str


class DraftViolation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str | None = None
    line: int | None
    claim_id: str
    claim_status: str | None
    violation_type: str
    severity: DraftSeverity
    message: str
    suggested_rewrite: str | None = None
    suppressed_by_override: bool = False
    window: str = ""
    matched_phrase: str | None = None


class DraftCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    violations: list[DraftViolation] = Field(default_factory=list)
    overrides: list[DraftOverride] = Field(default_factory=list)
    passed_claim_ids: list[str] = Field(default_factory=list)

    @property
    def unsuppressed_blocking(self) -> list[DraftViolation]:
        return [
            item
            for item in self.violations
            if item.severity == DraftSeverity.BLOCKING and not item.suppressed_by_override
        ]

    def to_text(self) -> str:
        lines: list[str] = []
        for violation in self.violations:
            suppressed = " suppressed" if violation.suppressed_by_override else ""
            lines.append(
                f"{violation.severity.value.upper()} {violation.file}:{violation.line} "
                f"[CLAIM:{violation.claim_id}] {violation.violation_type}{suppressed}"
            )
            lines.append(violation.message)
            if violation.suggested_rewrite:
                lines.append(f"Suggested fix: {violation.suggested_rewrite}")
        for override in self.overrides:
            lines.append(
                f"OVERRIDE {override.file}:{override.line} [CLAIM:{override.claim_id}] "
                f"{override.violation_type}: {override.reason}"
            )
        return "\n".join(lines) + ("\n" if lines else "Draft Guard: no blocking findings.\n")

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def check_draft_files(
    files: list[Path],
    *,
    claim_ledger_path: Path,
    allow_overrides: bool = True,
) -> DraftCheckResult:
    ledger = parse_claim_ledger(claim_ledger_path)
    combined = DraftCheckResult()
    for path in files:
        result = check_text(path.read_text(encoding="utf-8"), ledger, file=str(path))
        if not allow_overrides:
            for violation in result.violations:
                violation.suppressed_by_override = False
        combined.violations.extend(result.violations)
        combined.overrides.extend(result.overrides)
        combined.passed_claim_ids.extend(result.passed_claim_ids)
    return combined


def check_text(text: str, ledger: ClaimLedger, *, file: str | None = None) -> DraftCheckResult:
    result = DraftCheckResult()
    for malformed in MALFORMED_TAG_PATTERN.finditer(text):
        result.violations.append(
            DraftViolation(
                file=file,
                line=_line(text, malformed.start()),
                claim_id="UNKNOWN",
                claim_status=None,
                violation_type="malformed_claim_tag",
                severity=DraftSeverity.BLOCKING,
                message=f"Malformed claim tag: {malformed.group(0)}",
                suggested_rewrite="Use [CLAIM:C001], \\claim{C001}, or <!-- CLAIM:C001 -->.",
            )
        )
    for tag in find_tags(text):
        claim = ledger.claims.get(tag.claim_id)
        if claim is None:
            result.violations.append(
                DraftViolation(
                    file=file,
                    line=tag.line,
                    claim_id=tag.claim_id,
                    claim_status=None,
                    violation_type="unknown_claim",
                    severity=DraftSeverity.BLOCKING,
                    message=f"Unknown claim ID {tag.claim_id}.",
                    suggested_rewrite="add the claim to research/logs/claim_ledger.md.",
                )
            )
            continue
        window = extract_sentence_window(text, tag.start)
        paragraph = extract_paragraph(text, tag.start)
        overrides = overrides_for_paragraph(paragraph, text, tag, file=file)
        result.overrides.extend(overrides)
        local_violations: list[DraftViolation] = []
        for phrase in claim.forbidden:
            if phrase_matches(window, phrase) and not phrase_is_negated(window, phrase):
                local_violations.append(
                    DraftViolation(
                        file=file,
                        line=tag.line,
                        claim_id=tag.claim_id,
                        claim_status=claim.status.value,
                        violation_type="forbidden_language",
                        severity=DraftSeverity.BLOCKING,
                        message=f"Forbidden claim language matched: {phrase!r}.",
                        suggested_rewrite=(
                            "use scoped language from the claim ledger `allowed` list."
                        ),
                        window=window,
                        matched_phrase=phrase,
                    )
                )
        for caveat in claim.required_caveats:
            if not phrase_matches(window, caveat):
                local_violations.append(
                    DraftViolation(
                        file=file,
                        line=tag.line,
                        claim_id=tag.claim_id,
                        claim_status=claim.status.value,
                        violation_type="missing_required_caveat",
                        severity=DraftSeverity.WARNING,
                        message=f"Required caveat missing near claim tag: {caveat!r}.",
                        suggested_rewrite=f"include the caveat `{caveat}` near the claim tag.",
                        window=window,
                    )
                )
        for debt in claim.debt_flags:
            local_violations.append(
                DraftViolation(
                    file=file,
                    line=tag.line,
                    claim_id=tag.claim_id,
                    claim_status=claim.status.value,
                    violation_type="unresolved_scientific_debt",
                    severity=DraftSeverity.WARNING,
                    message=f"Claim has unresolved scientific debt flag: {debt}.",
                    suggested_rewrite="add a visible caveat or resolve/waive the debt by decision.",
                    window=window,
                )
            )
        _apply_overrides(local_violations, overrides)
        result.violations.extend(local_violations)
        if not local_violations:
            result.passed_claim_ids.append(tag.claim_id)
    return result


def find_tags(text: str) -> list[DraftTag]:
    tags: list[DraftTag] = []
    for match in TAG_PATTERN.finditer(text):
        claim_id = match.group("md_id") or match.group("tex_id") or match.group("html_id")
        tag_type = (
            "markdown" if match.group("markdown") else "latex" if match.group("latex") else "html"
        )
        tags.append(
            DraftTag(
                claim_id=claim_id,
                raw=match.group(0),
                start=match.start(),
                end=match.end(),
                line=_line(text, match.start()),
                tag_type=tag_type,
            )
        )
    return tags


def extract_sentence_window(text: str, offset: int, radius: int = 2) -> str:
    paragraph, paragraph_start = extract_paragraph_with_start(text, offset)
    local_offset = offset - paragraph_start
    sentences = list(SENTENCE.finditer(paragraph))
    if not sentences:
        return paragraph
    containing = next(
        (
            index
            for index, match in enumerate(sentences)
            if match.start() <= local_offset <= match.end()
        ),
        None,
    )
    if containing is None:
        return paragraph
    start = max(0, containing - radius)
    end = min(len(sentences), containing + radius + 1)
    return " ".join(match.group(0).strip() for match in sentences[start:end])


def phrase_matches(text: str, phrase: str) -> bool:
    words = re.findall(r"[A-Za-z0-9_]+", phrase.lower())
    if len(words) < 2:
        return False
    pattern = r"\b" + r"\W+".join(re.escape(word) for word in words) + r"\b"
    return re.search(pattern, text.lower(), flags=re.IGNORECASE | re.MULTILINE) is not None


def phrase_is_negated(text: str, phrase: str) -> bool:
    lower = text.lower()
    words = re.findall(r"[A-Za-z0-9_]+", phrase.lower())
    pattern = r"\b" + r"\W+".join(re.escape(word) for word in words) + r"\b"
    for match in re.finditer(pattern, lower, flags=re.IGNORECASE | re.MULTILINE):
        prefix = lower[max(0, match.start() - 240) : match.start()]
        if any(negation.search(prefix) for negation in NEGATED_CLAIM_PATTERNS):
            return True
    return False


def extract_paragraph(text: str, offset: int) -> str:
    return extract_paragraph_with_start(text, offset)[0]


def extract_paragraph_with_start(text: str, offset: int) -> tuple[str, int]:
    start = text.rfind("\n\n", 0, offset)
    end = text.find("\n\n", offset)
    start = 0 if start == -1 else start + 2
    end = len(text) if end == -1 else end
    return text[start:end], start


def overrides_for_paragraph(
    paragraph: str,
    full_text: str,
    tag: DraftTag,
    *,
    file: str | None,
) -> list[DraftOverride]:
    overrides: list[DraftOverride] = []
    paragraph_start = full_text.find(paragraph)
    for regex in (MARKDOWN_OVERRIDE, LATEX_OVERRIDE):
        for match in regex.finditer(paragraph):
            reason = match.group("reason").strip()
            if not reason:
                continue
            overrides.append(
                DraftOverride(
                    file=file,
                    line=_line(full_text, paragraph_start + match.start()),
                    claim_id=tag.claim_id,
                    violation_type=match.group("violation"),
                    reason=reason,
                    raw_comment=match.group(0),
                )
            )
    return overrides


def _apply_overrides(violations: list[DraftViolation], overrides: list[DraftOverride]) -> None:
    override_types = {override.violation_type for override in overrides}
    for violation in violations:
        if violation.violation_type in override_types:
            violation.suppressed_by_override = True


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1
