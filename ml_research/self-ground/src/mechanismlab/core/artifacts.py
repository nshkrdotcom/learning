from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from mechanismlab.core.claims import ClaimSpec
from mechanismlab.core.experiments import ExperimentSpec
from mechanismlab.core.runs import RunManifest


class ArtifactContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "mechanismlab.artifact_contract.v1"
    required: list[str]
    optional: list[str] = Field(default_factory=list)
    typed_artifacts: dict[str, str] = Field(default_factory=dict)


ModelT = TypeVar("ModelT", bound=BaseModel)


def _load_mapping(path: Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]
        except Exception as exc:
            raise ValueError(
                f"{path} is not JSON and PyYAML is not installed for YAML loading"
            ) from exc
        try:
            loaded = yaml.safe_load(text)
        except Exception as exc:
            raise ValueError(f"failed to parse {path} as JSON or YAML") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain an object") from None
        return loaded
    except Exception as exc:
        raise ValueError(f"failed to parse {path}") from exc


def _load_model(path: Path, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate(_load_mapping(Path(path)))


def load_claim_spec(path: Path) -> ClaimSpec:
    return _load_model(path, ClaimSpec)


def load_experiment_spec(path: Path) -> ExperimentSpec:
    return _load_model(path, ExperimentSpec)


def load_run_manifest(path: Path) -> RunManifest:
    return _load_model(path, RunManifest)


def write_model(model: BaseModel, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
