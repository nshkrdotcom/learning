from __future__ import annotations

from pathlib import Path
from typing import Any

from mwb.adapters.manifests import (
    AdapterCapabilityManifest,
    AdapterConformanceResult,
    ClaimBearingSupport,
    backend_version_manifest,
    package_version,
)
from mwb.domain.objects import ModelIdentity, TensorSpace
from mwb.hashing import sha256_text
from mwb.refs import stable_ref


class TransformerLensAdapter:
    adapter_name = "TransformerLensAdapter"
    adapter_version = "0.1.0"

    def capability_manifest(self) -> AdapterCapabilityManifest:
        return AdapterCapabilityManifest(
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            package={"name": "transformer_lens", "version": package_version("transformer-lens")},
            capabilities={
                "load_model": True,
                "run_with_cache": True,
                "capture_activation": True,
                "map_hook_to_tensor_space": True,
                "residual_stream_patch": True,
                "attention_head_patch": "conditional",
                "sae_feature_patch": False,
            },
            claim_bearing=ClaimBearingSupport(
                supported=True,
                required_conformance=[
                    "load_model_identity",
                    "hook_tensor_space_mapping",
                    "activation_capture_roundtrip",
                ],
            ),
            limitations=[
                "only TransformerLens-supported model architectures",
                "hook names follow TransformerLens conventions",
                "checkpoint hash availability depends on model source",
            ],
        )

    def backend_version_manifest(self, *, device: str) -> Any:
        return backend_version_manifest(
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            packages=["transformer_lens", "torch", "transformers"],
            device=device,
        )

    def model_identity_for_name(
        self,
        model_name: str,
        *,
        backend_version: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> ModelIdentity:
        config_hash = sha256_text(repr(sorted((config or {}).items()))) if config else None
        return ModelIdentity(
            wb_ref=stable_ref("model", "transformer_lens", model_name, backend_version or ""),
            provider="huggingface",
            model_name=model_name,
            backend="TransformerLens",
            backend_version=backend_version or package_version("transformer-lens"),
            tokenizer_ref=stable_ref("tok", model_name),
            config_hash=config_hash,
        )

    def tensor_space_for_hook(
        self,
        *,
        model_ref: str,
        hook_point: str,
        d_model: int | None,
    ) -> TensorSpace:
        return TensorSpace(
            wb_ref=stable_ref("space", model_ref, hook_point, d_model or "unknown"),
            model_ref=model_ref,
            hook_point=hook_point,
            axis_names=["batch", "position", "d_model"],
            dtype="float32",
            shape=[None, None, d_model],
        )

    def load_model(self, model_name: str, *, device: str):
        from transformer_lens import HookedTransformer

        return HookedTransformer.from_pretrained(model_name, device=device)

    def run_conformance(
        self,
        *,
        model_name: str,
        hook_point: str | None,
        device: str,
        output_dir: Path | None = None,
        dry_run: bool = False,
    ) -> AdapterConformanceResult:
        manifest = self.capability_manifest()
        versions = self.backend_version_manifest(device=device)
        result = AdapterConformanceResult(
            adapter_name=self.adapter_name,
            status="diagnostic_only" if dry_run else "pass",
            manifest=manifest.model_dump(mode="json"),
            backend_versions=versions.model_dump(mode="json"),
        )
        identity = self.model_identity_for_name(
            model_name,
            backend_version=versions.package_versions.get("transformer-lens"),
        )
        result.model_identity = identity.model_dump(mode="json")
        result.add_check("load_model_identity", "pass", model_ref=identity.wb_ref)

        if dry_run:
            result.add_check("dry_run_no_model_load", "pass")
            return result

        try:
            model = self.load_model(model_name, device=device)
            d_model = int(model.cfg.d_model)
            resolved_hook = hook_point or "blocks.0.hook_resid_post"
            space = self.tensor_space_for_hook(
                model_ref=identity.wb_ref,
                hook_point=resolved_hook,
                d_model=d_model,
            )
            result.tensor_space = space.model_dump(mode="json")
            result.add_check("hook_tensor_space_mapping", "pass", tensor_space_ref=space.wb_ref)

            tokens = model.to_tokens("The capital of France is")
            _, cache = model.run_with_cache(tokens, names_filter=lambda name: name == resolved_hook)
            activation = cache[resolved_hook]
            shape = [int(dim) for dim in activation.shape]
            result.add_check(
                "activation_capture_roundtrip",
                "pass",
                hook=resolved_hook,
                shape=shape,
            )
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "transformer_lens_conformance.json").write_text(
                    result.model_dump_json(indent=2),
                    encoding="utf-8",
                )
        except Exception as exc:
            result.status = "fail"
            result.errors.append(str(exc))
            result.add_check("adapter_exception", "fail", error=type(exc).__name__)
        return result
