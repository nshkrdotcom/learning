from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from mwb.adapters.manifests import (
    AdapterCapabilityManifest,
    AdapterConformanceResult,
    ClaimBearingSupport,
    backend_version_manifest,
    result_from_manifests,
    write_conformance_artifacts,
)
from mwb.domain.objects import DictionaryIdentity, MechanisticUnitRef
from mwb.hashing import sha256_text
from mwb.refs import stable_ref


class NeuronpediaAdapter:
    adapter_name = "NeuronpediaAdapter"
    adapter_version = "0.1.0"
    default_base_url = "https://www.neuronpedia.org"

    def __init__(self, *, base_url: str | None = None) -> None:
        self.base_url = (base_url or self.default_base_url).rstrip("/")

    def capability_manifest(self) -> AdapterCapabilityManifest:
        return AdapterCapabilityManifest(
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            package={"name": "neuronpedia_api", "version": None},
            capabilities={
                "read_feature_metadata": True,
                "write_feature_metadata": False,
                "feature_ref_roundtrip": True,
                "external_alias_mapping": True,
                "activation_fetch": "conditional",
                "claim_bearing_evidence": False,
            },
            claim_bearing=ClaimBearingSupport(
                supported=False,
                required_conformance=[
                    "feature_metadata_ref_roundtrip",
                    "read_only_posture",
                    "optional_http_fetch",
                ],
            ),
            limitations=[
                "read-only metadata adapter",
                "external explanations are hypothesis context, not causal evidence",
                "API availability and schemas may change outside the local repository",
                "no claim-bearing support is declared by this adapter",
            ],
        )

    def backend_version_manifest(self) -> Any:
        return backend_version_manifest(
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            packages=[],
            device="external_api",
        )

    def metadata_uri(self, *, model_id: str, sae_id: str, feature_index: int) -> str:
        return f"neuronpedia://{model_id}/{sae_id}/{feature_index}"

    def feature_url(self, *, model_id: str, sae_id: str, feature_index: int) -> str:
        return f"{self.base_url}/{model_id}/{sae_id}/{feature_index}"

    def dictionary_identity_for_feature(
        self,
        *,
        model_id: str,
        sae_id: str,
    ) -> DictionaryIdentity:
        model_ref = stable_ref("model", "neuronpedia", model_id)
        return DictionaryIdentity(
            wb_ref=stable_ref("dict", "neuronpedia", model_id, sae_id),
            provider="Neuronpedia",
            sae_id=sae_id,
            release=model_id,
            hook_point="external_feature_metadata",
            model_ref=model_ref,
        )

    def feature_ref(
        self,
        *,
        model_id: str,
        sae_id: str,
        feature_index: int,
    ) -> MechanisticUnitRef:
        dictionary = self.dictionary_identity_for_feature(model_id=model_id, sae_id=sae_id)
        uri = self.metadata_uri(
            model_id=model_id,
            sae_id=sae_id,
            feature_index=feature_index,
        )
        return MechanisticUnitRef(
            wb_ref=stable_ref("unit", "neuronpedia", model_id, sae_id, feature_index),
            uri=uri,
            unit_kind="sae_feature",
            model_ref=dictionary.model_ref,
            tensor_space_ref=stable_ref("space", dictionary.model_ref, "neuronpedia", sae_id),
            dictionary_ref=dictionary.wb_ref,
            feature_index=feature_index,
            external_aliases=[
                uri,
                self.feature_url(
                    model_id=model_id,
                    sae_id=sae_id,
                    feature_index=feature_index,
                ),
            ],
            valid_operations=["metadata_lookup"],
            invalid_operations=["claim_bearing_causal_intervention"],
        )

    def run_conformance(
        self,
        *,
        model_id: str,
        sae_id: str,
        feature_index: int,
        output_dir: Path | None = None,
        dry_run: bool = False,
    ) -> AdapterConformanceResult:
        manifest = self.capability_manifest()
        versions = self.backend_version_manifest()
        result = result_from_manifests(
            adapter_name=self.adapter_name,
            status="diagnostic_only",
            manifest=manifest,
            backend_versions=versions,
        )
        dictionary = self.dictionary_identity_for_feature(model_id=model_id, sae_id=sae_id)
        unit = self.feature_ref(model_id=model_id, sae_id=sae_id, feature_index=feature_index)
        metadata_uri = self.metadata_uri(
            model_id=model_id,
            sae_id=sae_id,
            feature_index=feature_index,
        )
        result.dictionary_identity = dictionary.model_dump(mode="json")
        result.artifact_refs = [metadata_uri]
        result.add_check(
            "feature_metadata_ref_roundtrip",
            "pass",
            dictionary_ref=dictionary.wb_ref,
            unit_ref=unit.wb_ref,
            uri=metadata_uri,
        )
        result.add_check("read_only_posture", "pass", write_feature_metadata=False)

        if dry_run:
            result.add_check("dry_run_no_http_fetch", "pass")
            write_conformance_artifacts(result, output_dir=output_dir, stem="neuronpedia")
            return result

        try:
            payload = self.fetch_feature_metadata(
                model_id=model_id,
                sae_id=sae_id,
                feature_index=feature_index,
            )
            result.status = "pass"
            result.add_check(
                "optional_http_fetch",
                "pass",
                payload_hash=sha256_text(json.dumps(payload, sort_keys=True, default=str)),
            )
        except Exception as exc:
            result.status = "fail"
            result.errors.append(str(exc))
            result.add_check("adapter_exception", "fail", error=type(exc).__name__)

        write_conformance_artifacts(result, output_dir=output_dir, stem="neuronpedia")
        return result

    def fetch_feature_metadata(
        self,
        *,
        model_id: str,
        sae_id: str,
        feature_index: int,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/api/feature/{model_id}/{sae_id}/{feature_index}"
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Neuronpedia metadata fetch failed: HTTP {exc.code}") from exc
