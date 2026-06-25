from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mechledger.core.claim_ledger import ClaimLedger
from mechledger.core.draft_severity import DraftSeverity
from mechledger.draftguard import extract_sentence_window, phrase_matches

CLAIM_TAG = re.compile(r"\[CLAIM:(?P<claim_id>C[0-9]+[A-Za-z0-9_-]*)\]")
SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)", re.MULTILINE)


class DraftFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    violation_type: str
    severity: DraftSeverity
    matched_phrase: str
    window: str
    line: int
    file: str | None = None


class DraftGuardProtoResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flagged: list[DraftFinding] = Field(default_factory=list)
    passed_claim_ids: list[str] = Field(default_factory=list)


def check_markdown_file(path: str | Path, ledger: ClaimLedger) -> DraftGuardProtoResult:
    path = Path(path)
    result = check_markdown_text(path.read_text(encoding="utf-8"), ledger)
    for finding in result.flagged:
        finding.file = str(path)
    return result


def check_markdown_text(text: str, ledger: ClaimLedger) -> DraftGuardProtoResult:
    result = DraftGuardProtoResult()
    for match in CLAIM_TAG.finditer(text):
        claim_id = match.group("claim_id")
        claim = ledger.claims.get(claim_id)
        if claim is None:
            result.flagged.append(
                DraftFinding(
                    claim_id=claim_id,
                    violation_type="unknown_claim",
                    severity=DraftSeverity.BLOCKING,
                    matched_phrase=claim_id,
                    window="",
                    line=_line(text, match.start()),
                )
            )
            continue
        window = extract_sentence_window(text, match.start())
        matched = next(
            (phrase for phrase in claim.forbidden if phrase_matches(window, phrase)), None
        )
        if matched:
            result.flagged.append(
                DraftFinding(
                    claim_id=claim_id,
                    violation_type="forbidden_language",
                    severity=DraftSeverity.BLOCKING,
                    matched_phrase=matched,
                    window=window,
                    line=_line(text, match.start()),
                )
            )
        else:
            result.passed_claim_ids.append(claim_id)
    return result


def _paragraph(text: str, offset: int) -> str:
    start = text.rfind("\n\n", 0, offset)
    end = text.find("\n\n", offset)
    start = 0 if start == -1 else start + 2
    end = len(text) if end == -1 else end
    return text[start:end]


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1
