from __future__ import annotations

import pytest

from construct_mismatch.tokenization import validate_single_token_targets


def test_invalid_multi_token_candidates_are_rejected() -> None:
    vocab = {" good": [10], " not_single": [11, 12]}

    def encode(text: str) -> list[int]:
        return vocab[text]

    def decode(token_ids) -> str:
        reverse = {(10,): " good", (11, 12): " not_single"}
        return reverse[tuple(token_ids)]

    with pytest.raises(ValueError, match="single-token mode"):
        validate_single_token_targets([" good", " not_single"], encode, decode)
