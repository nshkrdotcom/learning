from __future__ import annotations

import pytest
import torch

from self_ground.model import TransformerLensModelAdapter


class TinyTokenizerModel:
    def to_single_token(self, token: str) -> int:
        if token == " ok":
            return 7
        raise ValueError("not a single token")

    def to_tokens(self, token: str, prepend_bos: bool = False) -> torch.Tensor:
        if token == " multi":
            return torch.tensor([[1, 2]])
        return torch.tensor([[]], dtype=torch.long)


def test_token_ids_validate_single_token_strings_without_loading_model() -> None:
    adapter = TransformerLensModelAdapter.__new__(TransformerLensModelAdapter)
    adapter.model = TinyTokenizerModel()

    assert adapter.token_ids_for_strings([" ok"]) == [7]

    with pytest.raises(ValueError, match="exactly one token"):
        adapter.token_ids_for_strings([" multi"])

    with pytest.raises(ValueError, match="at least one"):
        adapter.token_ids_for_strings([])
