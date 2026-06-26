from __future__ import annotations

import hashlib
import json
import math
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from mechledger.alias import resolve_run_id
from mechledger.core.diagnostics import Diagnostic, DiagnosticSeverity
from mechledger.project import Project, now_utc

MUTABLE_PREDICTION_FIELDS = {
    "locked_at",
    "locked_content_hash",
    "scored_against_run_id",
    "sign_match",
    "relative_magnitude_match",
    "tamper_status",
}


class PredictionInputError(ValueError):
    pass


class PredictionStateError(RuntimeError):
    pass


class PredictedDirection(StrEnum):
    INCREASE = "increase"
    DECREASE = "decrease"
    NO_CHANGE = "no_change"
    UNKNOWN = "unknown"


class PredictedRelativeMagnitude(StrEnum):
    TARGET_GT_CONTROL = "target_gt_control"
    TARGET_LTE_CONTROL = "target_lte_control"
    UNKNOWN = "unknown"


class TamperStatus(StrEnum):
    NOT_LOCKED = "not_locked"
    LOCKED_VALID = "locked_valid"
    MODIFIED_AFTER_LOCK = "modified_after_lock"
    INVALIDATED = "invalidated"


class ExplainerPrediction(BaseModel):
    model_config = ConfigDict(extra="allow")

    prediction_id: str
    feature_id: str
    source_examples_path: str
    prediction_artifact_path: str
    label_source_model: str | None = None
    label_prompt_path: str | None = None
    label_generated_at: str | None = None
    short_label: str
    predicted_target_direction: PredictedDirection
    predicted_control_direction: PredictedDirection
    predicted_relative_magnitude: PredictedRelativeMagnitude
    locked_at: str | None = None
    locked_content_hash: str | None = None
    scored_against_run_id: str | None = None
    sign_match: bool | None = None
    relative_magnitude_match: bool | None = None
    tamper_status: TamperStatus = TamperStatus.NOT_LOCKED


def load_prediction(path: Path) -> ExplainerPrediction:
    payload = _load_prediction_payload(path)
    return _validate_payload(path, payload)


def write_prediction(path: Path, prediction: ExplainerPrediction) -> None:
    payload = prediction.model_dump(mode="json")
    _write_json_atomic(path, payload)


def canonical_prediction_hash(prediction_or_payload: ExplainerPrediction | dict[str, Any]) -> str:
    if isinstance(prediction_or_payload, ExplainerPrediction):
        payload = prediction_or_payload.model_dump(mode="json")
    else:
        payload = dict(prediction_or_payload)
    canonical_payload = _canonicalize_prediction_payload(payload)
    encoded = json.dumps(
        canonical_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def lock_prediction(path: Path, *, force: bool = False) -> ExplainerPrediction:
    payload = _load_prediction_payload(path)
    prediction = _validate_payload(path, payload)
    current_hash = canonical_prediction_hash(payload)
    if prediction.locked_content_hash:
        if prediction.locked_content_hash == current_hash and not force:
            if prediction.tamper_status != TamperStatus.LOCKED_VALID:
                prediction = prediction.model_copy(
                    update={"tamper_status": TamperStatus.LOCKED_VALID}
                )
                write_prediction(path, prediction)
            return prediction
        if prediction.locked_content_hash != current_hash and not force:
            prediction = prediction.model_copy(
                update={"tamper_status": TamperStatus.MODIFIED_AFTER_LOCK}
            )
            write_prediction(path, prediction)
            raise PredictionStateError(
                _prediction_state_message(
                    path,
                    prediction.prediction_id,
                    "prediction.modified_after_lock",
                    "Prediction semantic content changed after it was locked.",
                    "review the edit and run `mechledger prediction lock --force PATH` to relock.",
                )
            )
    prediction = prediction.model_copy(
        update={
            "locked_at": _fresh_lock_timestamp(prediction.locked_at),
            "locked_content_hash": current_hash,
            "scored_against_run_id": None,
            "sign_match": None,
            "relative_magnitude_match": None,
            "tamper_status": TamperStatus.LOCKED_VALID,
        }
    )
    write_prediction(path, prediction)
    return prediction


def find_prediction_by_id(
    project_root: Path,
    prediction_id: str,
    prediction_dirs: list[Path] | None = None,
) -> Path:
    paths = _prediction_search_paths(project_root, prediction_dirs)
    matches: list[Path] = []
    for path in paths:
        payload = _load_prediction_payload(path)
        raw_prediction_id = payload.get("prediction_id")
        if raw_prediction_id == prediction_id:
            matches.append(path)
    if len(matches) > 1:
        candidates = "\n".join(f"  {path}" for path in matches)
        raise PredictionInputError(
            f"Duplicate prediction_id `{prediction_id}` found in:\n{candidates}"
        )
    if not matches:
        raise PredictionInputError(f"Unknown prediction_id `{prediction_id}`.")
    return matches[0]


def score_prediction(
    project: Project,
    prediction_id: str,
    run_id_or_alias: str,
    *,
    prediction_dirs: list[Path] | None = None,
) -> ExplainerPrediction:
    path = find_prediction_by_id(project.root, prediction_id, prediction_dirs)
    return score_prediction_file(project, path, run_id_or_alias)


def score_prediction_file(
    project: Project,
    prediction_path: Path,
    run_id_or_alias: str,
) -> ExplainerPrediction:
    prediction = load_prediction(prediction_path)
    if not prediction.locked_content_hash or not prediction.locked_at:
        raise PredictionStateError(
            _prediction_state_message(
                prediction_path,
                prediction.prediction_id,
                "prediction.lock.required",
                f"Prediction {prediction.prediction_id} is not locked.",
                "run `mechledger prediction lock PATH` before scoring.",
            )
        )
    if prediction.tamper_status != TamperStatus.LOCKED_VALID:
        raise PredictionStateError(
            _prediction_state_message(
                prediction_path,
                prediction.prediction_id,
                "prediction.lock_state.invalid",
                (
                    f"Prediction {prediction.prediction_id} is not locked_valid "
                    f"({prediction.tamper_status})."
                ),
                "inspect tamper_status and relock only after human review.",
            )
        )
    current_hash = canonical_prediction_hash(prediction)
    if current_hash != prediction.locked_content_hash:
        prediction = prediction.model_copy(
            update={"tamper_status": TamperStatus.MODIFIED_AFTER_LOCK}
        )
        write_prediction(prediction_path, prediction)
        raise PredictionStateError(
            _prediction_state_message(
                prediction_path,
                prediction.prediction_id,
                "prediction.modified_after_lock",
                "Prediction semantic content changed after it was locked.",
                "review the edit and run `mechledger prediction lock --force PATH` to relock.",
            )
        )
    canonical_run_id = _resolve_existing_run(project, run_id_or_alias)
    run_dir = project.runs_dir / canonical_run_id
    run_data = _read_json_object(run_dir / "run.json")
    metric_rows = _read_jsonl_objects(run_dir / "metrics.jsonl")
    event_rows = _read_jsonl_objects(run_dir / "events.jsonl")
    observed_features = _observed_feature_ids(run_data, metric_rows, event_rows)
    if not observed_features:
        raise PredictionStateError(
            _prediction_state_message(
                prediction_path,
                prediction.prediction_id,
                "prediction.score.feature_evidence_missing",
                (
                    f"Target run {canonical_run_id} does not contain intervention evidence "
                    "with feature_id or features_modified metadata."
                ),
                "log feature_id or features_modified metadata in run.json, metrics, or events.",
            )
        )
    if prediction.feature_id not in observed_features:
        raise PredictionStateError(
            _prediction_state_message(
                prediction_path,
                prediction.prediction_id,
                "prediction.score.feature_mismatch",
                (
                    f"Prediction {prediction.prediction_id} feature ID mismatch: "
                    f"{prediction.feature_id} not in run {canonical_run_id} evidence "
                    f"{sorted(observed_features)}"
                ),
                "score against a run that declares the predicted feature in registered evidence.",
            )
        )
    metrics = _metrics_by_name(metric_rows)
    target_delta = _metric_value(metrics, ("target_delta", "top_target_delta"))
    control_delta = _metric_value(
        metrics,
        ("matched_control_delta", "top_matched_control_delta", "top_control_delta"),
    )
    if target_delta is None or control_delta is None:
        raise PredictionStateError(
            _prediction_state_message(
                prediction_path,
                prediction.prediction_id,
                "prediction.score.metric_missing",
                (
                    f"Run {canonical_run_id} is missing a required scoring metric: "
                    "target_delta/top_target_delta and matched_control_delta/"
                    "top_matched_control_delta/top_control_delta are required."
                ),
                "register the required run metrics before scoring this prediction.",
            )
        )
    specificity_gap = _metric_value(metrics, ("specificity_gap", "specificity_gap_mean"))
    if specificity_gap is not None and not math.isclose(
        specificity_gap,
        target_delta - control_delta,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise PredictionInputError(
            f"Run {canonical_run_id} has inconsistent specificity_gap: "
            f"{specificity_gap} != target_delta - matched_control_delta "
            f"({target_delta - control_delta})."
        )
    actual_target = _direction_from_delta(target_delta)
    actual_control = _direction_from_delta(control_delta)
    actual_relative = (
        PredictedRelativeMagnitude.TARGET_GT_CONTROL
        if target_delta > control_delta
        else PredictedRelativeMagnitude.TARGET_LTE_CONTROL
    )
    sign_match = _sign_match(
        prediction.predicted_target_direction,
        prediction.predicted_control_direction,
        actual_target,
        actual_control,
    )
    relative_magnitude_match = (
        None
        if prediction.predicted_relative_magnitude == PredictedRelativeMagnitude.UNKNOWN
        else prediction.predicted_relative_magnitude == actual_relative
    )
    prediction = prediction.model_copy(
        update={
            "scored_against_run_id": canonical_run_id,
            "sign_match": sign_match,
            "relative_magnitude_match": relative_magnitude_match,
            "tamper_status": TamperStatus.LOCKED_VALID,
        }
    )
    write_prediction(prediction_path, prediction)
    return prediction


def _validate_payload(path: Path, payload: dict[str, Any]) -> ExplainerPrediction:
    try:
        return ExplainerPrediction.model_validate(payload)
    except ValidationError as exc:
        raise PredictionInputError(_prediction_validation_message(path, payload, exc)) from exc


def _load_prediction_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PredictionInputError(
            _prediction_state_message(
                path,
                None,
                "prediction.file.missing",
                "Prediction file does not exist.",
                "check the path or create the prediction JSON file.",
            )
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PredictionInputError(
            _prediction_state_message(
                path,
                None,
                "prediction.json.invalid",
                f"Prediction file has malformed JSON: {exc.msg}",
                "fix the JSON syntax.",
            )
        ) from exc
    if not isinstance(payload, dict):
        raise PredictionInputError(
            _prediction_state_message(
                path,
                None,
                "prediction.json.type",
                "Prediction file must contain a JSON object.",
                "replace the JSON root with an object containing prediction_id and fields.",
            )
        )
    return payload


def _prediction_validation_message(
    path: Path,
    payload: dict[str, Any],
    exc: ValidationError,
) -> str:
    object_id = str(payload.get("prediction_id") or "<unknown>")
    failed_fields = [
        ".".join(str(item) for item in error.get("loc", ())) or "<root>"
        for error in exc.errors()
    ]
    details = "\n".join(
        f"- {field}: {error.get('msg')}"
        for field, error in zip(failed_fields, exc.errors(), strict=False)
    )
    diagnostic = Diagnostic(
        severity=DiagnosticSeverity.ERROR,
        code="prediction.validation",
        message=f"Invalid prediction record.\nFailed fields: {', '.join(failed_fields)}\n{details}",
        file=str(path),
        object_id=object_id,
        suggested_fix="make the prediction JSON match the ExplainerPrediction schema.",
    )
    return diagnostic.format()


def _prediction_state_message(
    path: Path,
    prediction_id: str | None,
    rule: str,
    message: str,
    suggested_fix: str,
) -> str:
    return Diagnostic(
        severity=DiagnosticSeverity.ERROR,
        code=rule,
        message=message,
        file=str(path),
        object_id=prediction_id,
        suggested_fix=suggested_fix,
    ).format()


def _canonicalize_prediction_payload(payload: dict[str, Any]) -> dict[str, Any]:
    stripped = {
        key: value
        for key, value in payload.items()
        if key not in MUTABLE_PREDICTION_FIELDS
    }
    return _sort_mappings(stripped)


def _sort_mappings(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sort_mappings(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sort_mappings(item) for item in value]
    return value


def _prediction_search_paths(
    project_root: Path,
    prediction_dirs: list[Path] | None,
) -> list[Path]:
    directories = [
        project_root / "research/predictions",
        project_root / "predictions",
    ]
    for directory in prediction_dirs or []:
        directories.append(_resolve_project_path(project_root, directory))
    paths: list[Path] = []
    seen: set[Path] = set()
    for directory in directories:
        if not directory.exists():
            continue
        if directory.is_file():
            candidates = [directory]
        else:
            candidates = sorted(directory.glob("**/*.json"))
        for path in candidates:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(resolved)
    return paths


def _resolve_project_path(project_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _resolve_existing_run(project: Project, run_id_or_alias: str) -> str:
    try:
        canonical = resolve_run_id(project, run_id_or_alias)
    except Exception as exc:  # noqa: BLE001
        raise PredictionInputError(str(exc)) from exc
    run_dir = project.runs_dir / canonical
    for name in ("run.json", "metrics.jsonl", "events.jsonl"):
        if not (run_dir / name).exists():
            raise PredictionInputError(f"Run {canonical} is missing required file: {name}")
    return canonical


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PredictionInputError(f"Malformed JSON in {path}") from exc
    if not isinstance(payload, dict):
        raise PredictionInputError(f"Expected JSON object in {path}")
    return payload


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PredictionInputError(f"Malformed JSONL in {path}:{line_number}") from exc
        if not isinstance(payload, dict):
            raise PredictionInputError(f"Expected JSON object in {path}:{line_number}")
        rows.append(payload)
    return rows


def _metrics_by_name(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for row in rows:
        metric_name = row.get("metric_name")
        if metric_name:
            metrics[str(metric_name)] = row.get("value")
    return metrics


def _metric_value(metrics: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = metrics.get(name)
        if value in (None, ""):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise PredictionInputError(f"Metric {name} must be numeric.") from exc
        if not math.isfinite(parsed):
            raise PredictionInputError(f"Metric {name} must be finite.")
        return parsed
    return None


def _observed_feature_ids(
    run_data: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
) -> set[str]:
    feature_ids: set[str] = set()
    _collect_feature_value(feature_ids, run_data.get("feature_id"))
    for row in metric_rows:
        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            _collect_feature_value(feature_ids, metadata.get("feature_id"))
    for row in event_rows:
        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            _collect_feature_value(feature_ids, metadata.get("feature_id"))
            _collect_feature_value(feature_ids, metadata.get("features_modified"))
    return feature_ids


def _collect_feature_value(feature_ids: set[str], value: Any) -> None:
    if value in (None, ""):
        return
    if isinstance(value, list | tuple | set):
        for item in value:
            _collect_feature_value(feature_ids, item)
        return
    feature_ids.add(str(value))


def _direction_from_delta(delta: float) -> PredictedDirection:
    if delta > 0.0:
        return PredictedDirection.INCREASE
    if delta < 0.0:
        return PredictedDirection.DECREASE
    return PredictedDirection.NO_CHANGE


def _sign_match(
    predicted_target: PredictedDirection,
    predicted_control: PredictedDirection,
    actual_target: PredictedDirection,
    actual_control: PredictedDirection,
) -> bool | None:
    if (
        predicted_target == PredictedDirection.UNKNOWN
        or predicted_control == PredictedDirection.UNKNOWN
    ):
        return None
    return predicted_target == actual_target and predicted_control == actual_control


def _fresh_lock_timestamp(previous: str | None) -> str:
    current = now_utc()
    if not previous or current != previous:
        return current
    try:
        from datetime import datetime

        parsed = datetime.fromisoformat(previous.replace("Z", "+00:00"))
    except ValueError:
        return current
    return (parsed + timedelta(seconds=1)).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)
