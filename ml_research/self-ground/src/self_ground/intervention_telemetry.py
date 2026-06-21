from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InterventionTelemetry:
    selected_feature_activation_mean: float
    selected_feature_activation_abs_mean: float
    selected_feature_modified_mean: float
    selected_feature_delta_abs_mean: float
    decoded_delta_norm_mean: float
    activation_norm_mean: float
    patched_activation_norm_mean: float
    relative_norm_drift_mean: float
    decoded_delta_norm_ratio: float

    def model_dump(self) -> dict[str, float]:
        return asdict(self)


def _finite(value: float) -> bool:
    return bool(np.isfinite(value))


def telemetry_has_nonfinite(telemetry: InterventionTelemetry | dict[str, Any]) -> bool:
    values = telemetry.model_dump() if isinstance(telemetry, InterventionTelemetry) else telemetry
    return any(not _finite(float(value)) for value in values.values() if value is not None)


def telemetry_warnings(
    telemetry: InterventionTelemetry | dict[str, Any],
    *,
    max_relative_norm_drift_warning: float = 0.5,
    max_decoded_delta_norm_ratio_warning: float = 0.5,
) -> dict[str, bool]:
    values = telemetry.model_dump() if isinstance(telemetry, InterventionTelemetry) else telemetry
    return {
        "norm_drift_warning": float(values.get("relative_norm_drift_mean", 0.0))
        > max_relative_norm_drift_warning,
        "decoded_delta_norm_ratio_warning": float(values.get("decoded_delta_norm_ratio", 0.0))
        > max_decoded_delta_norm_ratio_warning,
    }


def mean_telemetry(
    left: InterventionTelemetry | dict[str, Any],
    right: InterventionTelemetry | dict[str, Any],
) -> dict[str, float]:
    left_values = left.model_dump() if isinstance(left, InterventionTelemetry) else left
    right_values = right.model_dump() if isinstance(right, InterventionTelemetry) else right
    keys = set(left_values) | set(right_values)
    return {
        key: (float(left_values[key]) + float(right_values[key])) / 2.0
        for key in keys
        if key in left_values and key in right_values
    }
