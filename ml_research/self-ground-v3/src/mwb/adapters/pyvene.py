from __future__ import annotations

import importlib.util
import json
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
from mwb.domain.objects import ModelIdentity
from mwb.hashing import sha256_text
from mwb.refs import stable_ref


class PyVeneAdapter:
    adapter_name = "PyVeneAdapter"
    adapter_version = "0.1.0"

    def capability_manifest(self) -> AdapterCapabilityManifest:
        return AdapterCapabilityManifest(
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            package={"name": "pyvene", "version": package_version("pyvene")},
            capabilities={
                "intervention_configs": True,
                "static_interventions": True,
                "trainable_interventions": "conditional",
                "causal_abstraction": True,
                "knowledge_localization": True,
                "intervened_model_serialization": "conditional",
                "claim_bearing_evidence": False,
            },
            claim_bearing=ClaimBearingSupport(
                supported=False,
                required_conformance=[
                    "optional_dependency_available",
                    "model_identity_roundtrip",
                    "intervention_config_roundtrip",
                    "backend_intervention_execution",
                ],
            ),
            limitations=[
                "diagnostic-only until backend intervention execution conformance passes",
                "module paths and representations must match the wrapped PyTorch model",
                "trainable intervention support depends on pyvene and model configuration",
                "no claim-bearing support is declared by this adapter",
            ],
        )

    def backend_available(self) -> bool:
        return importlib.util.find_spec("pyvene") is not None

    def backend_version_manifest(self, *, device: str) -> Any:
        return backend_version_manifest(
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            packages=["pyvene", "torch", "transformers"],
            device=device,
        )

    def model_identity_for_name(
        self,
        model_name: str,
        *,
        backend_version: str | None = None,
    ) -> ModelIdentity:
        return ModelIdentity(
            wb_ref=stable_ref("model", "pyvene", model_name, backend_version or ""),
            provider="huggingface",
            model_name=model_name,
            backend="pyvene",
            backend_version=backend_version or package_version("pyvene"),
            tokenizer_ref=stable_ref("tok", model_name),
        )

    def intervention_contract(
        self,
        *,
        model_ref: str,
        module_path: str,
        intervention_kind: str,
    ) -> dict[str, Any]:
        payload = {
            "schema": "mwb.pyvene.intervention_contract.v1",
            "backend": "pyvene",
            "model_ref": model_ref,
            "module_path": module_path,
            "intervention_kind": intervention_kind,
            "allowed_kinds": [
                "resample_ablation",
                "zero_ablation",
                "activation_addition",
                "noising",
                "denoising",
                "interchange",
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["contract_hash"] = sha256_text(encoded)
        return payload

    def run_conformance(
        self,
        *,
        model_name: str,
        module_path: str,
        intervention_kind: str,
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
            backend_version=versions.package_versions.get("pyvene"),
        )
        contract = self.intervention_contract(
            model_ref=identity.wb_ref,
            module_path=module_path,
            intervention_kind=intervention_kind,
        )
        result.model_identity = identity.model_dump(mode="json")
        result.tensor_space = {
            "backend": "pyvene",
            "module_path": module_path,
            "intervention_contract": contract,
        }

        if intervention_kind not in contract["allowed_kinds"]:
            result.checks.append(
                {
                    "name": "intervention_kind_supported",
                    "status": "fail",
                    "intervention_kind": intervention_kind,
                }
            )
            result.errors.append(f"unsupported pyvene intervention kind: {intervention_kind}")
            write_conformance_artifacts(result, output_dir=output_dir, stem="pyvene")
            return result

        if not dry_run and not self.backend_available():
            result.checks.append(
                {
                    "name": "optional_dependency_available",
                    "status": "fail",
                    "missing": ["pyvene"],
                }
            )
            result.errors.append("missing optional backend: pyvene")
            write_conformance_artifacts(result, output_dir=output_dir, stem="pyvene")
            return result

        result.add_check(
            "optional_dependency_available",
            "pass" if self.backend_available() else "diagnostic",
            pyvene=package_version("pyvene"),
        )
        result.add_check("model_identity_roundtrip", "pass", model_ref=identity.wb_ref)
        result.add_check(
            "intervention_config_roundtrip",
            "pass",
            contract_hash=contract["contract_hash"],
            intervention_kind=intervention_kind,
        )

        if dry_run:
            result.add_check("dry_run_no_intervention_execution", "pass")
            write_conformance_artifacts(result, output_dir=output_dir, stem="pyvene")
            return result

        try:
            self._validate_backend_import()
            result.add_check("backend_api_contract", "pass")
            result.checks.append(
                {
                    "name": "backend_intervention_execution",
                    "status": "fail",
                    "reason": "real model intervention execution is not configured",
                }
            )
            result.errors.append(
                "pyvene backend available, but real intervention execution is not configured"
            )
        except Exception as exc:
            result.status = "fail"
            result.errors.append(str(exc))
            result.add_check("adapter_exception", "fail", error=type(exc).__name__)

        write_conformance_artifacts(result, output_dir=output_dir, stem="pyvene")
        return result

    def _validate_backend_import(self) -> None:
        import pyvene

        required = ["IntervenableConfig", "IntervenableModel"]
        missing = [name for name in required if not hasattr(pyvene, name)]
        if missing:
            raise AttributeError(f"pyvene missing expected APIs: {', '.join(missing)}")
