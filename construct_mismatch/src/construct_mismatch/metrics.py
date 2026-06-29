from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def signed_logit_diff(logit_diff: float | np.ndarray, labels: list[str] | np.ndarray) -> np.ndarray:
    values = np.asarray(logit_diff, dtype=float)
    label_array = np.asarray(labels)
    signs = np.where(label_array == "class_a", 1.0, -1.0)
    return values * signs


def accuracy_from_signed(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(np.asarray(values) > 0.0))


def standard_error(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = np.mean(sample)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def kl_divergence_from_logits(reference_logits: torch.Tensor, shifted_logits: torch.Tensor) -> float:
    ref_log_probs = F.log_softmax(reference_logits, dim=-1)
    shifted_log_probs = F.log_softmax(shifted_logits, dim=-1)
    ref_probs = ref_log_probs.exp()
    kl = F.kl_div(shifted_log_probs, ref_probs, reduction="batchmean", log_target=False)
    return float(kl.detach().cpu())
