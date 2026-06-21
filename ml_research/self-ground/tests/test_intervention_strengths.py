from __future__ import annotations

import pytest

from self_ground.real_behavioral_intervention import parse_amplify_factors, parse_operations


def test_operation_and_factor_parsing() -> None:
    assert parse_operations("ablate,amplify") == ["ablate", "amplify"]
    assert parse_amplify_factors("1.5,2.0") == [1.5, 2.0]


def test_noop_and_negative_amplify_factors_are_rejected() -> None:
    with pytest.raises(ValueError, match="1.0"):
        parse_amplify_factors("1.0")
    with pytest.raises(ValueError, match="positive"):
        parse_amplify_factors("-2.0")


def test_invalid_operation_is_rejected() -> None:
    with pytest.raises(ValueError, match="operations"):
        parse_operations("ablate,zero")
