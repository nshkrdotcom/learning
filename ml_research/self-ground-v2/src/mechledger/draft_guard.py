from __future__ import annotations

import re
from pathlib import Path

from mechledger.models import (
    ClaimLedger,
    DraftCheckResult,
    DraftOverride,
    DraftReference,
    DraftViolation,
)

TAG_RE = re.compile(
    r"(?P<markdown>\[CLAIM:(?P<markdown_id>[^\]]*)\])|"
    r"(?P<latex>\\claim\{(?P<latex_id>[^}]*)\})|"
    r"(?P<html><!--\s*CLAIM:(?P<html_id>[^-<\s]+)\s*-->)"
)
OVERRIDE_RE = re.compile(
    r"(?P<raw><!--\s*mechledger-disable\s+(?P<type>[a-z_]+):\s*(?P<reason>.*?)\s*-->|"
    r"%\s*mechledger-disable\s+(?P<tex_type>[a-z_]+):\s*(?P<tex_reason>.*?)(?:\n|$))",
    re.IGNORECASE | re.DOTALL,
)
SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)", re.MULTILINE)


def check_draft_file(path: str | Path, ledger: ClaimLedger) -> DraftCheckResult:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    references: list[DraftReference] = []
    violations: list[DraftViolation] = []
    overrides: list[DraftOverride] = []
    for match in TAG_RE.finditer(text):
        claim_id = (
            match.group("markdown_id") or match.group("latex_id") or match.group("html_id") or ""
        ).strip()
        line = _line_for_offset(text, match.start())
        raw_tag = match.group(0)
        if not claim_id:
            violations.append(
                DraftViolation(
                    file=path,
                    line=line,
                    claim_id="",
                    claim_status=None,
                    violation_type="malformed_claim_tag",
                    severity="blocking",
                    message="Malformed claim tag; expected a non-empty claim ID.",
                )
            )
            continue
        reference = DraftReference(
            file=path, line=line, claim_id=claim_id, raw_tag=raw_tag, offset=match.start()
        )
        references.append(reference)
        paragraph = _paragraph_containing(text, match.start())
        found_overrides = _overrides_for_paragraph(path, paragraph, claim_id, text)
        overrides.extend(found_overrides)
        if claim_id not in ledger.claims:
            violations.append(
                DraftViolation(
                    file=path,
                    line=line,
                    claim_id=claim_id,
                    claim_status=None,
                    violation_type="unknown_claim",
                    severity="blocking",
                    message=f"Claim ID {claim_id} is not present in {ledger.path}.",
                )
            )
            continue
        claim = ledger.claims[claim_id]
        window = _sentence_window(text, match.start())
        for phrase in claim.forbidden:
            if _phrase_matches(window, phrase):
                violations.append(
                    DraftViolation(
                        file=path,
                        line=line,
                        claim_id=claim_id,
                        claim_status=claim.status,
                        violation_type="forbidden_language",
                        severity="blocking",
                        message=f"Forbidden phrase near {claim_id}: {phrase!r}",
                        suggested_rewrite=_suggest_rewrite(claim),
                    )
                )
                break
        missing_caveats = [
            caveat for caveat in claim.required_caveats if not _phrase_matches(window, caveat)
        ]
        if missing_caveats:
            violations.append(
                DraftViolation(
                    file=path,
                    line=line,
                    claim_id=claim_id,
                    claim_status=claim.status,
                    violation_type="missing_required_caveat",
                    severity="warning",
                    message=f"Missing required caveat(s): {', '.join(missing_caveats)}",
                    suggested_rewrite=_suggest_rewrite(claim),
                )
            )
        if claim.debt_flags:
            violations.append(
                DraftViolation(
                    file=path,
                    line=line,
                    claim_id=claim_id,
                    claim_status=claim.status,
                    violation_type="unresolved_scientific_debt",
                    severity="warning",
                    message="Claim has unresolved scientific debt: " + ", ".join(claim.debt_flags),
                    suggested_rewrite=_suggest_rewrite(claim),
                )
            )

    _apply_overrides(violations, overrides)
    return DraftCheckResult(
        file=path, references=references, violations=violations, overrides=overrides
    )


def _line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _paragraph_containing(text: str, offset: int) -> str:
    start = text.rfind("\n\n", 0, offset)
    end = text.find("\n\n", offset)
    if start == -1:
        start = 0
    else:
        start += 2
    if end == -1:
        end = len(text)
    return text[start:end]


def _overrides_for_paragraph(
    path: Path, paragraph: str, claim_id: str, full_text: str
) -> list[DraftOverride]:
    overrides: list[DraftOverride] = []
    paragraph_offset = full_text.find(paragraph)
    for match in OVERRIDE_RE.finditer(paragraph):
        violation_type = match.group("type") or match.group("tex_type") or ""
        reason = (match.group("reason") or match.group("tex_reason") or "").strip()
        if not violation_type or not reason:
            continue
        absolute = paragraph_offset + match.start() if paragraph_offset >= 0 else 0
        overrides.append(
            DraftOverride(
                file=path,
                line=_line_for_offset(full_text, absolute),
                claim_id=claim_id,
                violation_type=violation_type,
                reason=reason,
                raw_comment=match.group("raw"),
            )
        )
    return overrides


def _sentence_window(text: str, offset: int, radius: int = 2) -> str:
    sentences = list(SENTENCE_RE.finditer(text))
    if not sentences:
        return _paragraph_containing(text, offset)
    containing = None
    for index, match in enumerate(sentences):
        if match.start() <= offset <= match.end():
            containing = index
            break
    if containing is None:
        return _paragraph_containing(text, offset)
    start = max(0, containing - radius)
    end = min(len(sentences), containing + radius + 1)
    return " ".join(match.group(0) for match in sentences[start:end])


def _phrase_matches(text: str, phrase: str) -> bool:
    words = re.findall(r"[A-Za-z0-9_]+", phrase.lower())
    if not words:
        return False
    if len(words) == 1 and len(words[0]) < 8:
        return False
    pattern = r"\b" + r"\W+".join(re.escape(word) for word in words) + r"\b"
    return re.search(pattern, text.lower(), flags=re.IGNORECASE | re.MULTILINE) is not None


def _suggest_rewrite(claim) -> str | None:
    if claim.required_caveats:
        caveats = ", ".join(claim.required_caveats)
        return f"Use scoped language such as: preliminary evidence ({caveats})."
    if claim.allowed:
        return f"Use one of the allowed terms where accurate: {', '.join(claim.allowed)}."
    return None


def _apply_overrides(violations: list[DraftViolation], overrides: list[DraftOverride]) -> None:
    for violation in violations:
        for override in overrides:
            if (
                override.claim_id == violation.claim_id
                and override.violation_type == violation.violation_type
            ):
                violation.suppressed_by_override = True
