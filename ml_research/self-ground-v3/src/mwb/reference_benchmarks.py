from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from mwb.domain.objects import ReferenceBenchmarkReport, ReferenceTask
from mwb.project import Project
from mwb.refs import stable_ref
from mwb.sqlite_index import initialize_schema, insert_payload

JsonDict = dict[str, Any]
DEFAULT_FDR_ALPHA = 0.1
DEFAULT_EXACT_EFFECT_THRESHOLD = 0.25
LOW_NULL_EFFECTS = [
    0.02,
    -0.02,
    0.03,
    -0.01,
    0.04,
    -0.03,
    0.01,
    -0.02,
    0.02,
    -0.01,
    0.03,
    -0.02,
    0.04,
    -0.03,
    0.01,
    -0.02,
    0.02,
    -0.01,
    0.03,
    -0.02,
]


class ReferenceBenchmarkService:
    def __init__(self, project: Project | None = None) -> None:
        self.project = project
        self.registry = {"toy": _toy_suite}

    def run_suite(self, suite: str = "toy") -> ReferenceBenchmarkReport:
        loader = self.registry.get(suite)
        if loader is None:
            expected = ", ".join(sorted(self.registry))
            raise ValueError(
                f"unknown reference benchmark suite {suite!r}; expected one of: {expected}"
            )
        tasks = [task.model_dump(mode="json") for task in loader()]
        scored_tasks = [_score_task(task) for task in tasks]
        calibration = _calibration(scored_tasks)
        summary = {
            "task_count": len(scored_tasks),
            "passed_count": sum(1 for task in scored_tasks if task["passed"]),
            "failed_count": sum(1 for task in scored_tasks if not task["passed"]),
            "classification_counts": _classification_counts(scored_tasks),
        }
        blockers = [
            f"reference_task_failed:{task['task_id']}"
            for task in scored_tasks
            if not task["passed"]
        ]
        status = "pass" if not blockers else "fail"
        return ReferenceBenchmarkReport(
            wb_ref=stable_ref("bench", suite, scored_tasks, calibration, status),
            suite=suite,
            status=status,
            tasks=scored_tasks,
            summary=summary,
            calibration=calibration,
            blockers=blockers,
            parents=[task["wb_ref"] for task in tasks],
        )

    def write_report(self, report: ReferenceBenchmarkReport) -> Path:
        if self.project is None:
            raise ValueError("ReferenceBenchmarkService.write_report requires a project")
        output_dir = self.project.mechanism_dir / "benchmarks"
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = report.model_dump(mode="json")
        latest = output_dir / "latest_framework_benchmark.json"
        stable = output_dir / f"{report.wb_ref}.json"
        _write_json(latest, payload)
        _write_json(stable, payload)
        initialize_schema(self.project.sqlite_path)
        insert_payload(self.project.sqlite_path, "benchmark_reports", report.wb_ref, payload)
        for task in report.tasks:
            insert_payload(
                self.project.sqlite_path,
                "reference_tasks",
                str(task["task_ref"]),
                task,
            )
        return latest


def _toy_suite() -> list[ReferenceTask]:
    return [
        ReferenceTask(
            wb_ref=stable_ref("reftask", "toy", "toy_residual_sign"),
            suite="toy",
            task_id="toy_residual_sign",
            task_kind="planted_residual_direction",
            ground_truth={
                "mechanism_units": ["unit_direct_writer"],
                "effect_sign": "positive",
                "description": "A planted residual direction directly writes the target logit.",
            },
            fixture={
                "candidates": [
                    {
                        "unit_ref": "unit_direct_writer",
                        "proxy_score": 0.86,
                        "exact_effect": 0.74,
                        "null_effects": LOW_NULL_EFFECTS,
                    },
                    {
                        "unit_ref": "unit_context_reader",
                        "proxy_score": 0.42,
                        "exact_effect": 0.13,
                        "null_effects": LOW_NULL_EFFECTS,
                    },
                    {
                        "unit_ref": "unit_position_marker",
                        "proxy_score": 0.31,
                        "exact_effect": 0.04,
                        "null_effects": [0.05, -0.05, 0.04, 0.01, -0.02, 0.03],
                    },
                ]
            },
        ),
        ReferenceTask(
            wb_ref=stable_ref("reftask", "toy", "negative_control_surface_confound"),
            suite="toy",
            task_id="negative_control_surface_confound",
            task_kind="negative_control_confound",
            ground_truth={
                "mechanism_units": ["unit_semantic_writer"],
                "confound_units": ["unit_surface_token"],
                "description": "A surface token feature has high proxy score but no causal effect.",
            },
            fixture={
                "candidates": [
                    {
                        "unit_ref": "unit_surface_token",
                        "proxy_score": 0.91,
                        "exact_effect": 0.03,
                        "null_effects": [0.04, -0.03, 0.02, 0.01, -0.02, 0.03],
                    },
                    {
                        "unit_ref": "unit_semantic_writer",
                        "proxy_score": 0.37,
                        "exact_effect": 0.52,
                        "null_effects": LOW_NULL_EFFECTS,
                    },
                ]
            },
        ),
        ReferenceTask(
            wb_ref=stable_ref("reftask", "toy", "synthetic_sae_split_absorption"),
            suite="toy",
            task_id="synthetic_sae_split_absorption",
            task_kind="synthetic_sae_dictionary",
            ground_truth={
                "split_latents": ["negation"],
                "absorbed_features": ["feature_absorbed_negation_sentiment"],
                "description": (
                    "Synthetic dictionary contains a split latent and an absorbed feature."
                ),
            },
            fixture={
                "features": [
                    {
                        "feature_ref": "feature_negation_split_a",
                        "latent_weights": {"negation": 0.53},
                    },
                    {
                        "feature_ref": "feature_negation_split_b",
                        "latent_weights": {"negation": 0.51},
                    },
                    {
                        "feature_ref": "feature_absorbed_negation_sentiment",
                        "latent_weights": {"negation": 0.48, "sentiment": 0.46},
                    },
                    {
                        "feature_ref": "feature_clean_sentiment",
                        "latent_weights": {"sentiment": 0.81},
                    },
                ]
            },
        ),
    ]


def _score_task(task: JsonDict) -> JsonDict:
    task_kind = str(task["task_kind"])
    if task_kind == "planted_residual_direction":
        scored = _score_planted_mechanism(task)
    elif task_kind == "negative_control_confound":
        scored = _score_negative_control(task)
    elif task_kind == "synthetic_sae_dictionary":
        scored = _score_synthetic_sae(task)
    else:
        scored = {
            "classification": "unknown_reference_task",
            "passed": False,
            "blockers": ["unknown_reference_task"],
            "checks": [],
            "scores": {},
        }
    return {
        "task_ref": task["wb_ref"],
        "suite": task["suite"],
        "task_id": task["task_id"],
        "task_kind": task_kind,
        "ground_truth": task["ground_truth"],
        **scored,
    }


def _score_planted_mechanism(task: JsonDict) -> JsonDict:
    candidates = _candidate_scores(task)
    expected = set(task["ground_truth"].get("mechanism_units", []))
    significant = {
        row["unit_ref"]
        for row in candidates
        if row["q_value"] <= DEFAULT_FDR_ALPHA
        and abs(row["exact_effect"]) >= DEFAULT_EXACT_EFFECT_THRESHOLD
    }
    top_exact = _top(candidates, "exact_effect")
    found = sorted(expected & significant)
    passed = bool(found) and top_exact["unit_ref"] in expected
    return {
        "classification": "mechanism_found" if passed else "known_mechanism_missed",
        "passed": passed,
        "found_units": found,
        "blockers": [] if passed else ["known_mechanism_missed"],
        "checks": [
            {
                "name": "known_unit_recovery",
                "status": "pass" if passed else "fail",
                "metrics": {"significant_units": sorted(significant)},
            }
        ],
        "scores": _score_summary(candidates),
        "candidate_scores": candidates,
    }


def _score_negative_control(task: JsonDict) -> JsonDict:
    candidates = _candidate_scores(task)
    confounds = set(task["ground_truth"].get("confound_units", []))
    top_proxy = _top(candidates, "proxy_score")
    top_exact = _top(candidates, "exact_effect")
    proxy_is_confound = top_proxy["unit_ref"] in confounds
    confound_blocked = (
        proxy_is_confound
        and top_proxy["q_value"] > DEFAULT_FDR_ALPHA
        and abs(top_proxy["exact_effect"]) < DEFAULT_EXACT_EFFECT_THRESHOLD
    )
    passed = bool(confound_blocked and top_exact["unit_ref"] != top_proxy["unit_ref"])
    return {
        "classification": "false_positive_blocked" if passed else "false_positive_leaked",
        "passed": passed,
        "found_units": [top_exact["unit_ref"]] if passed else [],
        "blockers": ["tempting_confound"] if passed else ["false_positive_leaked"],
        "checks": [
            {
                "name": "negative_control_rejection",
                "status": "pass" if passed else "fail",
                "metrics": {
                    "top_proxy_unit": top_proxy["unit_ref"],
                    "top_proxy_q_value": top_proxy["q_value"],
                    "top_proxy_exact_effect": top_proxy["exact_effect"],
                },
            }
        ],
        "scores": _score_summary(candidates),
        "candidate_scores": candidates,
    }


def _score_synthetic_sae(task: JsonDict) -> JsonDict:
    features = list(task["fixture"].get("features", []))
    split_latents = _detect_split_latents(features)
    absorbed_features = _detect_absorbed_features(features)
    expected_splits = list(task["ground_truth"].get("split_latents", []))
    expected_absorbed = list(task["ground_truth"].get("absorbed_features", []))
    split_passed = sorted(split_latents) == sorted(expected_splits)
    absorbed_passed = sorted(absorbed_features) == sorted(expected_absorbed)
    passed = split_passed and absorbed_passed
    return {
        "classification": (
            "dictionary_artifact_detected" if passed else "dictionary_artifact_missed"
        ),
        "passed": passed,
        "found_units": sorted({*split_latents, *absorbed_features}),
        "blockers": [] if passed else ["dictionary_artifact_missed"],
        "checks": [
            {
                "name": "feature_split",
                "status": "pass" if split_passed else "fail",
                "metrics": {"split_latents": sorted(split_latents)},
            },
            {
                "name": "feature_absorption",
                "status": "pass" if absorbed_passed else "fail",
                "metrics": {"absorbed_features": sorted(absorbed_features)},
            },
        ],
        "scores": {
            "split_latent_count": len(split_latents),
            "absorbed_feature_count": len(absorbed_features),
        },
        "candidate_scores": [],
    }


def _candidate_scores(task: JsonDict) -> list[JsonDict]:
    candidates = list(task["fixture"].get("candidates", []))
    p_values = [_empirical_p_value(candidate) for candidate in candidates]
    q_values = _benjamini_hochberg(p_values)
    rows = []
    for candidate, p_value, q_value in zip(candidates, p_values, q_values, strict=True):
        rows.append(
            {
                "unit_ref": str(candidate["unit_ref"]),
                "proxy_score": float(candidate["proxy_score"]),
                "exact_effect": float(candidate["exact_effect"]),
                "p_value": p_value,
                "q_value": q_value,
                "null_effects": [float(value) for value in candidate.get("null_effects", [])],
            }
        )
    return rows


def _empirical_p_value(candidate: JsonDict) -> float:
    exact = abs(float(candidate["exact_effect"]))
    nulls = [abs(float(value)) for value in candidate.get("null_effects", [])]
    if not nulls:
        return 1.0
    exceedances = sum(1 for value in nulls if value >= exact)
    return (exceedances + 1) / (len(nulls) + 1)


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1], reverse=True)
    q_values = [1.0] * len(p_values)
    previous = 1.0
    total = len(p_values)
    for rank_from_end, (index, p_value) in enumerate(indexed, start=1):
        rank = total - rank_from_end + 1
        q_value = min(previous, p_value * total / rank)
        previous = q_value
        q_values[index] = min(q_value, 1.0)
    return q_values


def _detect_split_latents(features: list[JsonDict]) -> list[str]:
    by_latent: dict[str, list[tuple[str, float]]] = {}
    for feature in features:
        weights = feature.get("latent_weights", {})
        strong_weights = {
            str(latent): float(weight)
            for latent, weight in weights.items()
            if float(weight) >= 0.45
        }
        if len(strong_weights) != 1:
            continue
        for latent, raw_weight in weights.items():
            weight = float(raw_weight)
            if weight >= 0.45:
                by_latent.setdefault(str(latent), []).append((str(feature["feature_ref"]), weight))
    return [
        latent
        for latent, rows in by_latent.items()
        if len(rows) >= 2 and sum(weight for _feature, weight in rows) >= 0.9
    ]


def _detect_absorbed_features(features: list[JsonDict]) -> list[str]:
    absorbed = []
    for feature in features:
        weights = feature.get("latent_weights", {})
        strong_latents = [latent for latent, weight in weights.items() if float(weight) >= 0.35]
        if len(strong_latents) >= 2:
            absorbed.append(str(feature["feature_ref"]))
    return absorbed


def _calibration(tasks: list[JsonDict]) -> JsonDict:
    candidates = [
        candidate
        for task in tasks
        for candidate in task.get("candidate_scores", [])
        if "proxy_score" in candidate and "exact_effect" in candidate
    ]
    proxy = [float(candidate["proxy_score"]) for candidate in candidates]
    exact = [abs(float(candidate["exact_effect"])) for candidate in candidates]
    nulls = [
        abs(float(value))
        for candidate in candidates
        for value in candidate.get("null_effects", [])
    ]
    q_values = [float(candidate["q_value"]) for candidate in candidates]
    significant_q = [q_value for q_value in q_values if q_value <= DEFAULT_FDR_ALPHA]
    return {
        "proxy_vs_exact_correlation": _pearson(proxy, exact),
        "fdr_adjusted_p_value": min(q_values) if q_values else 1.0,
        "significant_candidate_count": len(significant_q),
        "null_seed_count": len(nulls),
        "empirical_null_abs_p95": _percentile(nulls, 0.95),
        "fdr_alpha": DEFAULT_FDR_ALPHA,
    }


def _score_summary(candidates: list[JsonDict]) -> JsonDict:
    top_proxy = _top(candidates, "proxy_score")
    top_exact = _top(candidates, "exact_effect")
    return {
        "top_proxy_unit": top_proxy["unit_ref"],
        "top_exact_unit": top_exact["unit_ref"],
        "top_proxy_score": top_proxy["proxy_score"],
        "top_exact_effect": top_exact["exact_effect"],
        "min_q_value": min(row["q_value"] for row in candidates),
    }


def _top(candidates: list[JsonDict], field: str) -> JsonDict:
    if field == "exact_effect":
        return max(candidates, key=lambda row: abs(float(row[field])))
    return max(candidates, key=lambda row: float(row[field]))


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / (denom_x * denom_y)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _classification_counts(tasks: list[JsonDict]) -> JsonDict:
    counts: dict[str, int] = {}
    for task in tasks:
        classification = str(task["classification"])
        counts[classification] = counts.get(classification, 0) + 1
    return counts


def _write_json(path: Path, payload: JsonDict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
