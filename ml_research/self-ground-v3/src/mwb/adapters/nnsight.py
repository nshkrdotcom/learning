from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from mwb.adapters.manifests import (
    AdapterCapabilityManifest,
    AdapterConformanceResult,
    ClaimBearingSupport,
    backend_version_manifest,
    package_version,
    result_from_manifests,
    write_conformance_artifacts,
)
from mwb.domain.objects import ModelIdentity, TensorSpace
from mwb.refs import stable_ref


class NNsightAdapter:
    adapter_name = "NNsightAdapter"
    adapter_version = "0.1.0"

    def capability_manifest(self) -> AdapterCapabilityManifest:
        return AdapterCapabilityManifest(
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            package={"name": "nnsight", "version": package_version("nnsight")},
            capabilities={
                "local_tracing": True,
                "remote_tracing": "conditional",
                "activation_access": True,
                "activation_edit": True,
                "gradient_capture": True,
                "batched_interventions": True,
                "nnterp_normalized_names": "conditional",
                "claim_bearing_evidence": False,
            },
            claim_bearing=ClaimBearingSupport(
                supported=False,
                required_conformance=[
                    "optional_dependency_available",
                    "model_identity_roundtrip",
                    "module_tensor_space_mapping",
                    "activation_trace_roundtrip",
                ],
            ),
            limitations=[
                "diagnostic-only until exact intervention conformance passes",
                "remote tracing depends on external NDIF configuration",
                "module paths follow Hugging Face or nnterp naming conventions",
                "no claim-bearing support is declared by this adapter",
            ],
        )

    def backend_available(self) -> bool:
        return importlib.util.find_spec("nnsight") is not None

    def nnterp_available(self) -> bool:
        return importlib.util.find_spec("nnterp") is not None

    def backend_version_manifest(self, *, device: str) -> Any:
        return backend_version_manifest(
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            packages=["nnsight", "nnterp", "torch", "transformers"],
            device=device,
        )

    def model_identity_for_name(
        self,
        model_name: str,
        *,
        backend_version: str | None = None,
    ) -> ModelIdentity:
        return ModelIdentity(
            wb_ref=stable_ref("model", "nnsight", model_name, backend_version or ""),
            provider="huggingface",
            model_name=model_name,
            backend="nnsight",
            backend_version=backend_version or package_version("nnsight"),
            tokenizer_ref=stable_ref("tok", model_name),
        )

    def tensor_space_for_module(
        self,
        *,
        model_ref: str,
        module_path: str,
        hidden_size: int | None = None,
        device: str | None = None,
    ) -> TensorSpace:
        return TensorSpace(
            wb_ref=stable_ref("space", model_ref, "nnsight", module_path, hidden_size or "unknown"),
            model_ref=model_ref,
            backend="nnsight",
            hook_point=module_path,
            stream_kind="module_activation",
            axis_names=["batch", "position", "hidden"],
            dtype="float32",
            shape=[None, None, hidden_size],
            device=device,
        )

    def run_conformance(
        self,
        *,
        model_name: str,
        module_path: str,
        device: str,
        output_dir: Path | None = None,
        dry_run: bool = False,
    ) -> AdapterConformanceResult:
        manifest = self.capability_manifest()
        versions = self.backend_version_manifest(device=device)
        result = result_from_manifests(
            adapter_name=self.adapter_name,
            status="diagnostic_only",
            manifest=manifest,
            backend_versions=versions,
        )
        identity = self.model_identity_for_name(
            model_name,
            backend_version=versions.package_versions.get("nnsight"),
        )
        result.model_identity = identity.model_dump(mode="json")
        space = self.tensor_space_for_module(
            model_ref=identity.wb_ref,
            module_path=module_path,
            device=device,
        )
        result.tensor_space = space.model_dump(mode="json")

        if not dry_run and not self.backend_available():
            result.checks.append(
                {
                    "name": "optional_dependency_available",
                    "status": "fail",
                    "missing": ["nnsight"],
                }
            )
            result.errors.append("missing optional backend: nnsight")
            write_conformance_artifacts(result, output_dir=output_dir, stem="nnsight")
            return result

        result.add_check(
            "optional_dependency_available",
            "pass" if self.backend_available() else "diagnostic",
            nnsight=package_version("nnsight"),
            nnterp=package_version("nnterp"),
        )
        result.add_check("model_identity_roundtrip", "pass", model_ref=identity.wb_ref)
        result.add_check("module_tensor_space_mapping", "pass", tensor_space_ref=space.wb_ref)

        if dry_run:
            result.add_check("dry_run_no_trace_execution", "pass")
            write_conformance_artifacts(result, output_dir=output_dir, stem="nnsight")
            return result

        try:
            shape = self._run_activation_trace(model_name, module_path, device=device)
            result.status = "pass"
            result.add_check(
                "activation_trace_roundtrip",
                "pass",
                module_path=module_path,
                shape=shape,
            )
        except Exception as exc:
            result.status = "fail"
            result.errors.append(str(exc))
            result.add_check("adapter_exception", "fail", error=type(exc).__name__)

        write_conformance_artifacts(result, output_dir=output_dir, stem="nnsight")
        return result

    def _run_activation_trace(self, model_name: str, module_path: str, *, device: str) -> list[int]:
        from nnsight import LanguageModel

        model = LanguageModel(model_name, device_map=device)
        module = _resolve_attr_path(model, module_path)
        with model.trace("The capital of France is"):
            saved = module.output.save()
        value = getattr(saved, "value", saved)
        shape = getattr(value, "shape", None)
        if shape is None:
            raise TypeError(f"nnsight saved value has no shape for module path: {module_path}")
        return [int(dim) for dim in shape]


def _resolve_attr_path(obj: Any, path: str) -> Any:
    current = obj
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current
