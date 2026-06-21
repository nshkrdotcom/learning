from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from self_ground.io import write_config

PACKAGE_ATTEMPTS = [
    "sae_bench",
    "saebench",
    "sae_bench.evals.ravel",
    "sae_bench.evals.ravel.eval",
    "ravel",
]


class SAEBenchRavelProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "not_installed",
        "import_ok",
        "blocked_api_incompatible",
        "bridge_feasible",
        "ran_real_eval",
    ]
    packages_attempted: list[str]
    imported_modules: list[str]
    detected_entrypoints: list[str]
    can_accept_custom_dataset: bool | None
    can_accept_custom_sae: bool | None
    can_accept_precomputed_activations: bool | None
    can_compute_cause_isolation: bool | None
    blockers: list[dict[str, Any]]
    recommended_next_step: str


def _public_callables(module: ModuleType) -> list[tuple[str, Callable[..., Any]]]:
    callables: list[tuple[str, Callable[..., Any]]] = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        try:
            value = getattr(module, name)
        except Exception:
            continue
        if callable(value):
            callables.append((name, value))
    return callables


def _signature_words(callables: list[tuple[str, Callable[..., Any]]]) -> set[str]:
    words: set[str] = set()
    for name, value in callables:
        words.add(name.lower())
        try:
            signature = inspect.signature(value)
        except (TypeError, ValueError):
            continue
        for parameter in signature.parameters:
            words.add(parameter.lower())
    return words


def _has_any(words: set[str], candidates: set[str]) -> bool:
    return any(
        candidate in word or word in candidate
        for word in words
        for candidate in candidates
    )


def _detect_entrypoints(modules: list[ModuleType]) -> list[str]:
    detected: list[str] = []
    interesting = {"ravel", "eval", "evaluate", "run", "score", "dataset", "benchmark"}
    for module in modules:
        for name, value in _public_callables(module):
            lowered = name.lower()
            if any(token in lowered for token in interesting):
                detected.append(f"{module.__name__}.{name}")
            elif inspect.isclass(value) and any(token in lowered for token in interesting):
                detected.append(f"{module.__name__}.{name}")
    return sorted(set(detected))


def probe_saebench_ravel_bridge(
    *,
    out_dir: str | Path,
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> SAEBenchRavelProbeResult:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    write_config({"packages_attempted": PACKAGE_ATTEMPTS}, out_path / "config.json")

    imported: list[ModuleType] = []
    blockers: list[dict[str, Any]] = []
    for package in PACKAGE_ATTEMPTS:
        try:
            imported.append(importer(package))
        except Exception as exc:
            blockers.append(
                {
                    "package": package,
                    "exception_class": type(exc).__name__,
                    "exception_message": str(exc),
                }
            )

    imported_modules = sorted({module.__name__ for module in imported})
    if not imported:
        result = SAEBenchRavelProbeResult(
            status="not_installed",
            packages_attempted=list(PACKAGE_ATTEMPTS),
            imported_modules=[],
            detected_entrypoints=[],
            can_accept_custom_dataset=None,
            can_accept_custom_sae=None,
            can_accept_precomputed_activations=None,
            can_compute_cause_isolation=None,
            blockers=blockers,
            recommended_next_step=(
                "Install upstream SAEBench/RAVEL in an optional environment, then rerun "
                "`uv run python scripts/probe_saebench_ravel_bridge.py --out "
                f"{out_path}`."
            ),
        )
        _write_probe_artifacts(result, out_path)
        return result

    all_callables: list[tuple[str, Callable[..., Any]]] = []
    for module in imported:
        all_callables.extend(_public_callables(module))
    words = _signature_words(all_callables)
    detected_entrypoints = _detect_entrypoints(imported)

    can_accept_custom_dataset = _has_any(
        words,
        {"dataset", "datasets", "task", "tasks", "examples", "custom_dataset"},
    )
    can_accept_custom_sae = _has_any(words, {"sae", "autoencoder", "sparse_autoencoder"})
    can_accept_precomputed_activations = _has_any(
        words,
        {"activation", "activations", "feature_activations", "cache"},
    )
    can_compute_cause_isolation = _has_any(
        words,
        {"ravel", "cause", "isolation", "intervention", "intervene"},
    )
    feasible = (
        can_accept_custom_dataset
        and can_accept_custom_sae
        and can_accept_precomputed_activations
        and can_compute_cause_isolation
    )
    api_blockers = []
    if not can_accept_custom_dataset:
        api_blockers.append("No inspected callable clearly accepts custom dataset/task rows.")
    if not can_accept_custom_sae:
        api_blockers.append("No inspected callable clearly accepts a custom SAE object.")
    if not can_accept_precomputed_activations:
        api_blockers.append("No inspected callable clearly accepts precomputed activations.")
    if not can_compute_cause_isolation:
        api_blockers.append("No inspected callable clearly exposes RAVEL cause/isolation scoring.")
    blockers.extend({"api_blocker": blocker} for blocker in api_blockers)

    result = SAEBenchRavelProbeResult(
        status="bridge_feasible" if feasible else "blocked_api_incompatible",
        packages_attempted=list(PACKAGE_ATTEMPTS),
        imported_modules=imported_modules,
        detected_entrypoints=detected_entrypoints,
        can_accept_custom_dataset=can_accept_custom_dataset,
        can_accept_custom_sae=can_accept_custom_sae,
        can_accept_precomputed_activations=can_accept_precomputed_activations,
        can_compute_cause_isolation=can_compute_cause_isolation,
        blockers=blockers,
        recommended_next_step=(
            "Implement a thin SAEBench/RAVEL adapter against the detected entrypoint."
            if feasible
            else "Do not claim upstream integration. Inspect detected entrypoints and either "
            "add a small upstream-compatible adapter or document the API mismatch."
        ),
    )
    _write_probe_artifacts(result, out_path)
    return result


def _write_probe_artifacts(result: SAEBenchRavelProbeResult, out_path: Path) -> None:
    write_config(result.model_dump(mode="json"), out_path / "probe_result.json")
    readme = f"""# SAEBench/RAVEL Bridge Probe

- status: `{result.status}`
- imported modules: `{", ".join(result.imported_modules) or "none"}`

This is a bounded feasibility probe. It does not run a SELF-GROUND evaluation
through upstream SAEBench/RAVEL unless the upstream API is actually importable
and inspectable.

Recommended next step:

```text
{result.recommended_next_step}
```
"""
    (out_path / "README.md").write_text(readme, encoding="utf-8")
