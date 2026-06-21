from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NegationRavelScores:
    cause_score: float
    isolation_score: float
    cause_minus_isolation: float
    isolation_to_cause_ratio: float | None


def cause_score(target_absolute_delta: float) -> float:
    return float(target_absolute_delta)


def isolation_score(control_absolute_delta: float) -> float:
    return float(control_absolute_delta)


def cause_minus_isolation(target_absolute_delta: float, control_absolute_delta: float) -> float:
    return cause_score(target_absolute_delta) - isolation_score(control_absolute_delta)


def isolation_to_cause_ratio(
    target_absolute_delta: float,
    control_absolute_delta: float,
) -> float | None:
    cause = cause_score(target_absolute_delta)
    if cause == 0.0:
        return None
    return isolation_score(control_absolute_delta) / cause


def summarize_negation_ravel_scores(
    *,
    target_absolute_delta: float,
    control_absolute_delta: float,
) -> NegationRavelScores:
    return NegationRavelScores(
        cause_score=cause_score(target_absolute_delta),
        isolation_score=isolation_score(control_absolute_delta),
        cause_minus_isolation=cause_minus_isolation(
            target_absolute_delta,
            control_absolute_delta,
        ),
        isolation_to_cause_ratio=isolation_to_cause_ratio(
            target_absolute_delta,
            control_absolute_delta,
        ),
    )
