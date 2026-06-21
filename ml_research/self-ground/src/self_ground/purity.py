from __future__ import annotations

import re
from collections import Counter

NEGATION_RE = re.compile(r"\b(?:not|no|never|cannot|can't|isn't|doesn't|don't|won't|n't)\b")
DECOY_RE = re.compile(r"\b(?:often|sometimes|usually|occasionally)\b")
TOKEN_RE = re.compile(r"[a-z]+n't|[a-z]+|\d+")


def tokenize_for_purity(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _counter_overlap(a_tokens: list[str], b_tokens: list[str]) -> float:
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0

    a_counts = Counter(a_tokens)
    b_counts = Counter(b_tokens)
    shared = sum((a_counts & b_counts).values())
    total = sum((a_counts | b_counts).values())
    return shared / total if total else 1.0


def _length_ratio(a_tokens: list[str], b_tokens: list[str]) -> float:
    if not a_tokens and not b_tokens:
        return 1.0
    longest = max(len(a_tokens), len(b_tokens))
    if longest == 0:
        return 1.0
    return min(len(a_tokens), len(b_tokens)) / longest


def _changed_token_score(a_tokens: list[str], b_tokens: list[str]) -> float:
    a_counts = Counter(a_tokens)
    b_counts = Counter(b_tokens)
    changed = sum((a_counts - b_counts).values()) + sum((b_counts - a_counts).values())
    return max(0.0, 1.0 - min(changed, 10) / 10.0)


def _target_marker_score(pos: str, neg: str, para: str | None, decoy: str | None) -> float:
    if not pos and not neg and not para and not decoy:
        return 1.0

    checks = [
        bool(NEGATION_RE.search(pos.lower())),
        not bool(NEGATION_RE.search(neg.lower())),
    ]
    if para is not None:
        checks.append(bool(NEGATION_RE.search(para.lower())))
    if decoy is not None:
        lowered = decoy.lower()
        checks.append(not bool(NEGATION_RE.search(lowered)))
        checks.append(bool(DECOY_RE.search(lowered)))
    return sum(checks) / len(checks)


def _pair_surface_score(a: str, b: str) -> float:
    a_tokens = tokenize_for_purity(a)
    b_tokens = tokenize_for_purity(b)
    overlap = _counter_overlap(a_tokens, b_tokens)
    length = _length_ratio(a_tokens, b_tokens)
    changed = _changed_token_score(a_tokens, b_tokens)
    return (0.45 * overlap) + (0.30 * length) + (0.25 * changed)


def score_control_purity(
    x_pos: str,
    x_neg: str,
    *,
    x_para: str | None = None,
    x_decoy: str | None = None,
) -> float:
    """Score how well controls preserve everything except negation presence.

    This is a deliberately simple Tier A/B scorer. It is deterministic, safe on
    empty inputs, and combines surface overlap, length agreement, changed-token
    count, and marker sanity checks.
    """

    direct = _pair_surface_score(x_pos, x_neg)
    auxiliary_scores: list[float] = []
    if x_para is not None:
        auxiliary_scores.append(_pair_surface_score(x_pos, x_para))
    if x_decoy is not None:
        auxiliary_scores.append(_pair_surface_score(x_pos, x_decoy))

    auxiliary = sum(auxiliary_scores) / len(auxiliary_scores) if auxiliary_scores else direct
    marker = _target_marker_score(x_pos, x_neg, x_para, x_decoy)
    score = (0.62 * direct) + (0.23 * auxiliary) + (0.15 * marker)
    return round(max(0.0, min(1.0, score)), 6)
