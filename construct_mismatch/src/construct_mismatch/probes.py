from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def labels_to_binary(labels: list[str]) -> np.ndarray:
    return np.asarray([1 if label == "class_a" else 0 for label in labels], dtype=int)


def fit_logistic_probe(activations: np.ndarray, labels: list[str], max_iter: int = 1000) -> Pipeline:
    probe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=max_iter, solver="lbfgs")),
        ]
    )
    probe.fit(activations, labels_to_binary(labels))
    return probe


def probe_signed_scores(probe: Pipeline, activations: np.ndarray, labels: list[str]) -> np.ndarray:
    probs = probe.predict_proba(activations)[:, 1]
    centered = probs - 0.5
    signs = np.where(np.asarray(labels) == "class_a", 1.0, -1.0)
    return centered * signs
