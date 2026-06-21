from __future__ import annotations

from typing import Final

TRANSFORMER_LENS_BACKEND: Final = "transformer_lens"
SAE_LENS_BACKEND: Final = "sae_lens"
NEGATION_RAVEL_ADAPTER: Final = "negation_ravel_adapter"
SAEBENCH_RAVEL_BACKEND: Final = "saebench_ravel"
NNSIGHT_BACKEND: Final = "nnsight"
PYVENE_BACKEND: Final = "pyvene"
LEGACY_FEATURE_SPACE_PROXY: Final = "legacy_feature_space_proxy"
RESIDUAL_SMOKE_DIAGNOSTIC: Final = "transformer_lens_residual_smoke_diagnostic"

FORBIDDEN_ENGINE_BACKENDS: Final = {"self_ground_generic_engine"}
CLAIM_ELIGIBLE_BACKENDS: Final = {
    TRANSFORMER_LENS_BACKEND,
    NEGATION_RAVEL_ADAPTER,
    SAEBENCH_RAVEL_BACKEND,
}


def validate_engine_backend(engine_backend: str) -> str:
    if engine_backend in FORBIDDEN_ENGINE_BACKENDS:
        raise ValueError(
            "SELF-GROUND does not expose or accept a self_ground_generic_engine backend; "
            "use an external backend such as TransformerLens, SAELens, SAEBench/RAVEL, "
            "nnsight, or pyvene."
        )
    return engine_backend


def is_claim_eligible_backend(engine_backend: str) -> bool:
    return engine_backend in CLAIM_ELIGIBLE_BACKENDS
