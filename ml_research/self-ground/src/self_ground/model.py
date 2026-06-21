from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenBatch:
    texts: list[str]
    tokens: torch.Tensor


@dataclass(frozen=True)
class ActivationBatch:
    texts: list[str]
    layer: str
    activations: torch.Tensor


class TransformerLensModelAdapter:
    """Thin real-model adapter around TransformerLens HookedTransformer."""

    def __init__(self, model_name: str = "gpt2-small", device: str | None = None) -> None:
        from transformer_lens import HookedTransformer

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HookedTransformer.from_pretrained(model_name, device=self.device)
        self.model.eval()

    def tokenize(self, texts: list[str]) -> TokenBatch:
        tokens = self.model.to_tokens(texts)
        return TokenBatch(texts=texts, tokens=tokens)

    @torch.no_grad()
    def get_activations(self, texts: list[str], hook_point: str) -> torch.Tensor:
        _, cache = self.model.run_with_cache(texts, names_filter=[hook_point])
        if hook_point not in cache:
            raise ValueError(
                f"hook point {hook_point!r} was not captured for model {self.model_name!r}"
            )
        return cache[hook_point].detach()

    @torch.no_grad()
    def forward_with_cache(self, texts: list[str], layer: str) -> ActivationBatch:
        activations = self.get_activations(texts, hook_point=layer)
        return ActivationBatch(texts=texts, layer=layer, activations=activations)

    @torch.no_grad()
    def logits_for_texts(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("texts must contain at least one string")
        return self.model(texts).detach()

    def _single_token_id(self, token: str) -> int:
        if not token:
            raise ValueError("token string must be non-empty")
        try:
            return int(self.model.to_single_token(token))
        except Exception as exc:
            tokens = self.model.to_tokens(token, prepend_bos=False).flatten()
            if tokens.numel() != 1:
                raise ValueError(
                    f"expected {token!r} to map to exactly one token id; got {tokens.numel()}"
                ) from exc
            return int(tokens[-1])

    def token_ids_for_strings(self, strings: list[str]) -> list[int]:
        if not strings:
            raise ValueError("strings must contain at least one token string")
        return [self._single_token_id(token) for token in strings]

    @torch.no_grad()
    def logit_contrast(
        self,
        texts: list[str],
        positive: list[str],
        negative: list[str],
    ) -> torch.Tensor:
        logits = self.logits_for_texts(texts)[:, -1, :]
        pos_ids = self.token_ids_for_strings(positive)
        neg_ids = self.token_ids_for_strings(negative)
        pos_logits = logits[:, pos_ids].mean(dim=-1)
        neg_logits = logits[:, neg_ids].mean(dim=-1)
        return pos_logits - neg_logits

    @torch.no_grad()
    def get_logit_contrast(
        self,
        texts: list[str],
        positive_tokens: list[str],
        negative_tokens: list[str],
    ) -> torch.Tensor:
        return self.logit_contrast(
            texts,
            positive=positive_tokens,
            negative=negative_tokens,
        )

    def score_negation_behavior(self, text: str) -> float:
        contrast = self.logit_contrast(
            [text],
            positive=[" not", " no", " never"],
            negative=[" often", " always", " sometimes"],
        )
        return float(contrast[0].detach().cpu().item())
