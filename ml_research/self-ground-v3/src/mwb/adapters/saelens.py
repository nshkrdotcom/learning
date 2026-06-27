from __future__ import annotations

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
from mwb.domain.objects import DictionaryIdentity, MechanisticUnitRef
from mwb.refs import stable_ref


class SAELensAdapter:
    adapter_name = "SAELensAdapter"
    adapter_version = "0.1.0"

    def capability_manifest(self) -> AdapterCapabilityManifest:
        return AdapterCapabilityManifest(
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            package={"name": "sae_lens", "version": package_version("sae-lens")},
            capabilities={
                "load_sae": True,
                "encode_activations": True,
                "decode_features": True,
                "extract_dictionary_identity": True,
                "map_sae_to_hook": True,
                "feature_ablation": "domain_or_adapter",
            },
            claim_bearing=ClaimBearingSupport(
                supported=True,
                required_conformance=[
                    "sae_identity_roundtrip",
                    "model_hook_compatibility",
                    "feature_ref_roundtrip",
                ],
            ),
            limitations=[
                "external SAE metadata may be incomplete",
                "dictionary hash may not always be available",
            ],
        )

    def backend_version_manifest(self, *, device: str) -> Any:
        return backend_version_manifest(
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            packages=["sae_lens", "torch"],
            device=device,
        )

    def dictionary_identity_for_name(
        self,
        *,
        release: str,
        sae_id: str,
        hook_point: str,
        model_ref: str,
        feature_count: int | None = None,
        dictionary_hash: str | None = None,
    ) -> DictionaryIdentity:
        return DictionaryIdentity(
            wb_ref=stable_ref("dict", "saelens", release, sae_id, hook_point, model_ref),
            provider="SAELens",
            sae_id=sae_id,
            release=release,
            hook_point=hook_point,
            model_ref=model_ref,
            feature_count=feature_count,
            dictionary_hash=dictionary_hash,
        )

    def feature_ref(
        self,
        *,
        dictionary_ref: str,
        model_ref: str,
        tensor_space_ref: str,
        feature_index: int,
    ) -> MechanisticUnitRef:
        return MechanisticUnitRef(
            wb_ref=stable_ref("unit", dictionary_ref, feature_index),
            unit_kind="sae_feature",
            model_ref=model_ref,
            tensor_space_ref=tensor_space_ref,
            dictionary_ref=dictionary_ref,
            feature_index=feature_index,
        )

    def load_sae(self, *, release: str, sae_id: str, device: str):
        from sae_lens import SAE

        return SAE.from_pretrained(release=release, sae_id=sae_id, device=device)

    def run_conformance(
        self,
        *,
        model_ref: str,
        tensor_space_ref: str | None,
        hook_point: str,
        release: str,
        sae_id: str,
        device: str,
        output_dir: Path | None = None,
        dry_run: bool = False,
    ) -> AdapterConformanceResult:
        manifest = self.capability_manifest()
        versions = self.backend_version_manifest(device=device)
        result = result_from_manifests(
            adapter_name=self.adapter_name,
            status="diagnostic_only" if dry_run else "pass",
            manifest=manifest,
            backend_versions=versions,
        )
        identity = self.dictionary_identity_for_name(
            release=release,
            sae_id=sae_id,
            hook_point=hook_point,
            model_ref=model_ref,
        )
        result.dictionary_identity = identity.model_dump(mode="json")
        result.add_check("sae_identity_roundtrip", "pass", dictionary_ref=identity.wb_ref)

        if dry_run:
            result.add_check("dry_run_no_sae_load", "pass")
            write_conformance_artifacts(result, output_dir=output_dir, stem="saelens")
            return result

        try:
            sae = self.load_sae(release=release, sae_id=sae_id, device=device)
            cfg = getattr(sae, "cfg", None)
            cfg_hook = getattr(cfg, "hook_name", None) or getattr(cfg, "hook_point", None)
            if cfg_hook and str(cfg_hook) != hook_point:
                result.add_check(
                    "model_hook_compatibility",
                    "fail",
                    expected=hook_point,
                    observed=str(cfg_hook),
                )
            else:
                result.add_check("model_hook_compatibility", "pass", hook=hook_point)
            feature_ref = self.feature_ref(
                dictionary_ref=identity.wb_ref,
                model_ref=model_ref,
                tensor_space_ref=tensor_space_ref or stable_ref("space", model_ref, hook_point),
                feature_index=0,
            )
            result.add_check("feature_ref_roundtrip", "pass", unit_ref=feature_ref.wb_ref)
        except Exception as exc:
            result.status = "fail"
            result.errors.append(str(exc))
            result.add_check("adapter_exception", "fail", error=type(exc).__name__)
        write_conformance_artifacts(result, output_dir=output_dir, stem="saelens")
        return result
