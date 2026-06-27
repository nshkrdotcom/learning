from __future__ import annotations

import pytest

from local_mi_lab.tokens import decode_token, token_id_for_single_token


class FakeTokenizer:
    vocab = {" A": 1, " B": 2, " C": 3}
    inverse = {1: " A", 2: " B", 3: " C"}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text in self.vocab:
            return [self.vocab[text]]
        parts = [f" {part}" for part in text.strip().split()]
        ids = [self.vocab[part] for part in parts if part in self.vocab]
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.inverse[token_id] for token_id in ids)


def test_token_lookup_works_on_tiny_fake_tokenizer() -> None:
    tokenizer = FakeTokenizer()
    assert token_id_for_single_token(tokenizer, " A") == 1
    assert decode_token(tokenizer, 2) == " B"


def test_expected_token_validation_catches_missing_case() -> None:
    tokenizer = FakeTokenizer()
    with pytest.raises(ValueError, match="missing"):
        token_id_for_single_token(tokenizer, " Z")


def test_expected_token_validation_catches_multi_token_case() -> None:
    tokenizer = FakeTokenizer()
    with pytest.raises(ValueError, match="one token"):
        token_id_for_single_token(tokenizer, " A B")
