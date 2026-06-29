from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DirectionSet:
    directions: dict[int, np.ndarray]
    mean_a: dict[int, np.ndarray]
    mean_b: dict[int, np.ndarray]


def normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < eps:
        return np.zeros_like(vector)
    return vector / norm


def fit_diff_in_means(activations: dict[int, np.ndarray], labels: list[str]) -> DirectionSet:
    label_array = np.asarray(labels)
    directions: dict[int, np.ndarray] = {}
    mean_a: dict[int, np.ndarray] = {}
    mean_b: dict[int, np.ndarray] = {}
    for layer, acts in activations.items():
        a = acts[label_array == "class_a"]
        b = acts[label_array == "class_b"]
        if len(a) == 0 or len(b) == 0:
            raise ValueError("Both classes are required to fit a direction.")
        mean_a[layer] = a.mean(axis=0)
        mean_b[layer] = b.mean(axis=0)
        directions[layer] = normalize(mean_a[layer] - mean_b[layer])
    return DirectionSet(directions=directions, mean_a=mean_a, mean_b=mean_b)


def project(activations: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return activations @ direction


def signed_projection_scores(scores: np.ndarray, labels: list[str]) -> np.ndarray:
    signs = np.where(np.asarray(labels) == "class_a", 1.0, -1.0)
    return scores * signs


def shuffled_label_direction(
    activations: dict[int, np.ndarray],
    labels: list[str],
    seed: int = 0,
) -> DirectionSet:
    rng = np.random.default_rng(seed)
    shuffled = list(labels)
    rng.shuffle(shuffled)
    return fit_diff_in_means(activations, shuffled)


def random_direction_like(direction: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    random_vector = rng.normal(size=direction.shape)
    return normalize(random_vector) * np.linalg.norm(direction)
