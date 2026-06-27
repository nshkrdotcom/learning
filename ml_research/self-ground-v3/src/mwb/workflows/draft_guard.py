from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

CLAIM_RE = re.compile(r"\[CLAIM:([A-Za-z0-9_\-]+)\]")


def load_claim_cards(mechanism_dir: Path) -> dict[str, dict[str, Any]]:
    claims: dict[str, dict[str, Any]] = {}
    claims_dir = mechanism_dir / "claims"
    if not claims_dir.exists():
        return claims
    for path in claims_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        claim_ref = payload.get("claim_ref") or path.stem
        claims[str(claim_ref)] = payload
    return claims


def check_draft_text(text: str, claim_cards: dict[str, dict[str, Any]]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    for match in CLAIM_RE.finditer(text):
        claim_ref = match.group(1)
        card = claim_cards.get(claim_ref)
        sentence = _sentence_window(text, match.start(), match.end())
        prose_sentence = CLAIM_RE.sub("", sentence)
        if card is None:
            findings.append(
                {"claim_ref": claim_ref, "status": "missing_card", "sentence": sentence}
            )
            continue
        blocked_terms = [
            term
            for term in card.get("blocked_language", [])
            if term.lower() in prose_sentence.lower()
        ]
        if blocked_terms:
            findings.append(
                {
                    "claim_ref": claim_ref,
                    "status": "blocked",
                    "sentence": sentence,
                    "blocked_terms": blocked_terms,
                    "suggested_replacements": card.get("allowed_language", []),
                }
            )
        elif _missing_required_caveats(prose_sentence, card):
            findings.append(
                {
                    "claim_ref": claim_ref,
                    "status": "caveated",
                    "sentence": sentence,
                    "required_caveats": card.get("required_caveats", []),
                    "suggested_replacements": card.get("allowed_language", []),
                }
            )
        else:
            findings.append({"claim_ref": claim_ref, "status": "allowed", "sentence": sentence})
    statuses = [finding["status"] for finding in findings]
    if "blocked" in statuses:
        status = "blocked"
    elif "caveated" in statuses:
        status = "caveated"
    elif "missing_card" in statuses:
        status = "missing_card"
    elif not statuses:
        status = "unknown_claim"
    else:
        status = "allowed"
    return {"status": status, "findings": findings}


def _sentence_window(text: str, start: int, end: int) -> str:
    line_left = text.rfind("\n", 0, start)
    line_right = text.find("\n", end)
    left = 0 if line_left == -1 else line_left + 1
    right = len(text) if line_right == -1 else line_right
    return text[left:right].strip()


def _missing_required_caveats(sentence: str, card: dict[str, Any]) -> bool:
    required = [str(caveat).lower() for caveat in card.get("required_caveats", [])]
    if not required:
        return False
    lowered = sentence.lower()
    return any(caveat not in lowered for caveat in required)
