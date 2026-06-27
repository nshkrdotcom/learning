from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mwb.domain.objects import StaticCheckResult, StaticCompilationReport
from mwb.project import Project
from mwb.refs import stable_ref
from mwb.sqlite_index import initialize_schema, insert_payload

JsonDict = dict[str, Any]
CompilerCheck = Callable[[dict[str, Any], str], StaticCheckResult]

DEFAULT_PROJECTION_THRESHOLDS = {
    "pass_score_gte": 0.03,
    "warn_score_gt": 0.0,
}
DEFAULT_NEIGHBOR_THRESHOLDS = {
    "fail_cosine_gte": 0.8,
    "warn_cosine_gte": 0.6,
}
DEFAULT_DENSITY_MAX_RATIO = 1.5
EPSILON = 1e-12


def has_static_compiler_payload(payload: dict[str, Any]) -> bool:
    return _static_config(payload) is not None


class StaticCompiler:
    def __init__(self, project: Project | None = None) -> None:
        self.project = project
        self.registry: dict[str, CompilerCheck] = {
            "decoder_unembed_projection": self._decoder_unembed_projection,
            "neighbor_interference": self._neighbor_interference,
            "activation_density": self._activation_density,
        }

    def compile_file(self, path: Path) -> StaticCompilationReport:
        return self.compile_payload(json.loads(path.read_text(encoding="utf-8")))

    def compile_payload(self, payload: dict[str, Any]) -> StaticCompilationReport:
        hypothesis_ref = str(payload["wb_ref"])
        config = _static_config(payload)
        if config is None:
            return self._report(
                hypothesis_ref=hypothesis_ref,
                checks=[
                    _check_result(
                        hypothesis_ref,
                        "static_compiler_payload",
                        "fail",
                        blockers=["static_compiler_missing"],
                    )
                ],
            )

        requested_checks = list(config.get("checks") or self.registry)
        results: list[StaticCheckResult] = []
        for check_name in requested_checks:
            check = self.registry.get(str(check_name))
            if check is None:
                results.append(
                    _check_result(
                        hypothesis_ref,
                        str(check_name),
                        "fail",
                        blockers=["unknown_static_check"],
                    )
                )
                continue
            results.append(check(config, hypothesis_ref))
        return self._report(hypothesis_ref=hypothesis_ref, checks=results)

    def write_report(self, report: StaticCompilationReport) -> Path:
        if self.project is None:
            raise ValueError("StaticCompiler.write_report requires a project")
        output_dir = self.project.mechanism_dir / "static_compiler"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "latest_static_compile.json"
        payload = report.model_dump(mode="json")
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        initialize_schema(self.project.sqlite_path)
        insert_payload(
            self.project.sqlite_path,
            "static_compiler_reports",
            report.wb_ref,
            payload,
        )
        for check in report.checks:
            check_ref = str(check["wb_ref"])
            insert_payload(self.project.sqlite_path, "static_check_results", check_ref, check)
        return output

    def _decoder_unembed_projection(
        self,
        config: dict[str, Any],
        hypothesis_ref: str,
    ) -> StaticCheckResult:
        try:
            decoder = _vector(config.get("decoder_vector"), "decoder_vector")
            target_token_ids = [str(token) for token in config["target_token_ids"]]
            foil_token_ids = [str(token) for token in config["foil_token_ids"]]
            unembedding = config["unembedding"]
            target = _mean_vectors(
                [
                    _vector(unembedding[token], f"unembedding[{token}]")
                    for token in target_token_ids
                ]
            )
            foil = _mean_vectors(
                [
                    _vector(unembedding[token], f"unembedding[{token}]")
                    for token in foil_token_ids
                ]
            )
            contrast = [
                target_value - foil_value
                for target_value, foil_value in zip(target, foil, strict=True)
            ]
            score = _dot(_normalize_l2(decoder), _normalize_l2(contrast))
        except (KeyError, TypeError, ValueError) as exc:
            return _check_result(
                hypothesis_ref,
                "decoder_unembed_projection",
                "fail",
                metrics={"error": str(exc)},
                blockers=["preflight_failed"],
            )

        thresholds = {**DEFAULT_PROJECTION_THRESHOLDS, **config.get("projection_thresholds", {})}
        blockers: list[str] = []
        warnings: list[str] = []
        if score >= float(thresholds["pass_score_gte"]):
            status = "pass"
        elif score > float(thresholds["warn_score_gt"]):
            status = "warn"
            warnings.append("weak_decoder_unembed_projection")
        else:
            status = "fail"
            blockers.append("preflight_failed")

        return _check_result(
            hypothesis_ref,
            "decoder_unembed_projection",
            status,
            score=score,
            metrics={
                "score": score,
                "target_token_ids": [
                    int(token) if token.isdigit() else token for token in target_token_ids
                ],
                "foil_token_ids": [
                    int(token) if token.isdigit() else token for token in foil_token_ids
                ],
                "tensor_space_ref": config.get("tensor_space_ref"),
                "unembedding_space_ref": config.get("unembedding_space_ref"),
                "normalization": "l2_cosine",
                "transform_ref": config.get("transform_ref"),
                "transform_applied": bool(config.get("transform_ref")),
            },
            blockers=blockers,
            warnings=warnings,
        )

    def _neighbor_interference(
        self,
        config: dict[str, Any],
        hypothesis_ref: str,
    ) -> StaticCheckResult:
        dictionary = config.get("dictionary")
        if not isinstance(dictionary, dict):
            return _check_result(
                hypothesis_ref,
                "neighbor_interference",
                "warn",
                warnings=["dictionary_geometry_unavailable"],
            )
        vectors = dictionary.get("decoder_vectors")
        if not isinstance(vectors, dict):
            return _check_result(
                hypothesis_ref,
                "neighbor_interference",
                "warn",
                warnings=["dictionary_geometry_unavailable"],
            )

        feature_ref = str(dictionary.get("feature_id") or config.get("feature_id") or "")
        try:
            feature_vector = _vector(
                vectors.get(feature_ref) or config.get("decoder_vector"),
                "dictionary.feature_vector",
            )
        except (TypeError, ValueError) as exc:
            return _check_result(
                hypothesis_ref,
                "neighbor_interference",
                "fail",
                metrics={"error": str(exc)},
                blockers=["dictionary_interference"],
            )

        nearest_ref: str | None = None
        nearest_cosine = -1.0
        for ref, raw_vector in vectors.items():
            if str(ref) == feature_ref:
                continue
            cosine = abs(_cosine(feature_vector, _vector(raw_vector, f"dictionary[{ref}]")))
            if cosine > nearest_cosine:
                nearest_ref = str(ref)
                nearest_cosine = cosine

        if nearest_cosine < 0.0:
            nearest_cosine = 0.0

        thresholds = {**DEFAULT_NEIGHBOR_THRESHOLDS, **dictionary.get("thresholds", {})}
        blockers: list[str] = []
        warnings: list[str] = []
        if nearest_cosine >= float(thresholds["fail_cosine_gte"]):
            status = "fail"
            blockers.append("dictionary_interference")
        elif nearest_cosine >= float(thresholds["warn_cosine_gte"]):
            status = "warn"
            warnings.append("neighbor_feature_interference")
        else:
            status = "pass"

        return _check_result(
            hypothesis_ref,
            "neighbor_interference",
            status,
            score=nearest_cosine,
            metrics={
                "nearest_neighbor_ref": nearest_ref,
                "nearest_neighbor_cosine": nearest_cosine,
                "feature_ref": feature_ref,
                "threshold_fail_gte": float(thresholds["fail_cosine_gte"]),
                "threshold_warn_gte": float(thresholds["warn_cosine_gte"]),
            },
            blockers=blockers,
            warnings=warnings,
        )

    def _activation_density(
        self,
        config: dict[str, Any],
        hypothesis_ref: str,
    ) -> StaticCheckResult:
        density = config.get("activation_density")
        if not isinstance(density, dict):
            return _check_result(
                hypothesis_ref,
                "activation_density",
                "warn",
                warnings=["activation_density_unavailable"],
            )

        target = float(density.get("target", 0.0))
        control = float(density.get("control", 0.0))
        max_ratio = float(density.get("max_ratio", DEFAULT_DENSITY_MAX_RATIO))
        ratio = _symmetric_ratio(target, control)
        warnings: list[str] = []
        if ratio > max_ratio:
            status = "warn"
            warnings.append("activation_density_mismatch")
        else:
            status = "pass"
        return _check_result(
            hypothesis_ref,
            "activation_density",
            status,
            score=ratio,
            metrics={
                "target_density": target,
                "control_density": control,
                "density_ratio": ratio,
                "max_ratio": max_ratio,
            },
            warnings=warnings,
        )

    def _report(
        self,
        *,
        hypothesis_ref: str,
        checks: list[StaticCheckResult],
    ) -> StaticCompilationReport:
        check_payloads = [_check_payload(check) for check in checks]
        blockers = _dedupe(
            [
                blocker
                for check in checks
                for blocker in check.blockers
            ]
        )
        warnings = _dedupe(
            [
                warning
                for check in checks
                for warning in check.warnings
            ]
        )
        statuses = [check.status for check in checks]
        if "fail" in statuses:
            status = "fail"
            gate = "FAIL"
        elif "warn" in statuses:
            status = "warn"
            gate = "WEAK"
        else:
            status = "pass"
            gate = "PASS"
        return StaticCompilationReport(
            wb_ref=stable_ref(
                "static",
                hypothesis_ref,
                status,
                gate,
                json.dumps(check_payloads, sort_keys=True),
            ),
            hypothesis_ref=hypothesis_ref,
            status=status,
            plausibility_gate=gate,
            checks=check_payloads,
            blockers=blockers,
            warnings=warnings,
            parents=[hypothesis_ref],
        )


def _static_config(payload: dict[str, Any]) -> dict[str, Any] | None:
    config = payload.get("static_compiler")
    if isinstance(config, dict):
        return config
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict) and isinstance(metadata.get("static_compiler"), dict):
        return metadata["static_compiler"]
    return None


def _check_result(
    hypothesis_ref: str,
    check_name: str,
    status: str,
    *,
    score: float | None = None,
    metrics: dict[str, Any] | None = None,
    blockers: list[str] | None = None,
    warnings: list[str] | None = None,
) -> StaticCheckResult:
    metric_payload = dict(metrics or {})
    return StaticCheckResult(
        wb_ref=stable_ref(
            "staticcheck",
            hypothesis_ref,
            check_name,
            status,
            json.dumps(metric_payload, sort_keys=True),
            blockers or [],
            warnings or [],
        ),
        hypothesis_ref=hypothesis_ref,
        check_name=check_name,
        status=status,
        score=score,
        metrics=metric_payload,
        blockers=list(blockers or []),
        warnings=list(warnings or []),
    )


def _check_payload(result: StaticCheckResult) -> dict[str, Any]:
    payload = result.model_dump(mode="json")
    payload["name"] = payload.pop("check_name")
    payload.update(result.metrics)
    return payload


def _vector(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty numeric vector")
    vector = [float(item) for item in value]
    if not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{label} contains non-finite values")
    return vector


def _mean_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("cannot average an empty vector set")
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("vector dimensions do not match")
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimension)]


def _dot(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vector dimensions do not match")
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )


def _normalize_l2(vector: list[float]) -> list[float]:
    norm = math.sqrt(_dot(vector, vector))
    denominator = max(norm, EPSILON)
    return [value / denominator for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    return _dot(_normalize_l2(left), _normalize_l2(right))


def _symmetric_ratio(left: float, right: float) -> float:
    if left == 0.0 and right == 0.0:
        return 1.0
    denominator = min(abs(left), abs(right))
    if denominator <= EPSILON:
        return math.inf
    return max(abs(left), abs(right)) / denominator


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped
