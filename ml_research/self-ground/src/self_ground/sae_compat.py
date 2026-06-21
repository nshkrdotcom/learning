from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from self_ground.io import write_config
from self_ground.real_model_check import DEFAULT_CHECK_TEXTS


@dataclass(frozen=True)
class SAECompatibilityResult:
    model_name: str
    hook_point: str
    sae_release: str
    sae_id: str
    activation_shape: list[int]
    encoded_shape: list[int]
    decoded_shape: list[int]
    d_model: int
    d_sae: int
    compatible: bool
    status: str
    error: str | None = None

    def model_dump(self) -> dict:
        return asdict(self)


def _shape(value) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return []
    return [int(dim) for dim in shape]


def sae_shapes_are_compatible(activation_shape: list[int], decoded_shape: list[int]) -> bool:
    if len(activation_shape) != 3:
        return False
    if decoded_shape == activation_shape:
        return True
    if len(decoded_shape) == 2:
        return (
            decoded_shape[0] == activation_shape[0]
            and decoded_shape[1] == activation_shape[2]
        )
    return False


def sae_encoded_shape_is_compatible(
    activation_shape: list[int],
    encoded_shape: list[int],
) -> bool:
    if len(activation_shape) != 3 or len(encoded_shape) not in {2, 3}:
        return False
    if encoded_shape[0] != activation_shape[0]:
        return False
    if len(encoded_shape) == 3 and encoded_shape[1] != activation_shape[1]:
        return False
    return True


def verify_sae_compatibility(
    *,
    model_name: str = "EleutherAI/pythia-70m",
    hook_point: str = "blocks.2.hook_resid_post",
    sae_release: str = "",
    sae_id: str = "",
    device: str | None = "cpu",
    texts: list[str] | None = None,
    out: str | Path | None = None,
) -> SAECompatibilityResult:
    check_texts = texts or DEFAULT_CHECK_TEXTS
    if not sae_release or not sae_id:
        result = SAECompatibilityResult(
            model_name=model_name,
            hook_point=hook_point,
            sae_release=sae_release,
            sae_id=sae_id,
            activation_shape=[],
            encoded_shape=[],
            decoded_shape=[],
            d_model=0,
            d_sae=0,
            compatible=False,
            status="error",
            error="sae_release and sae_id are required",
        )
        if out is not None:
            write_config(result.model_dump(), out)
        return result
    activation_shape: list[int] = []
    encoded_shape: list[int] = []
    decoded_shape: list[int] = []
    d_model = 0
    d_sae = 0

    try:
        from self_ground.model import TransformerLensModelAdapter
        from self_ground.sae import SAELensAdapter

        model = TransformerLensModelAdapter(model_name=model_name, device=device)
        activations = model.get_activations(check_texts, hook_point=hook_point)
        activation_shape = _shape(activations)
        if len(activation_shape) != 3:
            raise ValueError(
                "activation must have shape [batch, position, d_model]; "
                f"got {activation_shape}"
            )
        d_model = activation_shape[-1]

        sae = SAELensAdapter.from_pretrained(
            release=sae_release,
            sae_id=sae_id,
            device=device or model.device,
        )
        encoded = sae.encode(activations)
        encoded_shape = _shape(encoded.values)
        if not sae_encoded_shape_is_compatible(activation_shape, encoded_shape):
            raise ValueError(
                "encoded SAE activations are not shape-compatible with hook activation: "
                f"{encoded_shape} vs {activation_shape}"
            )
        d_sae = encoded_shape[-1]

        decoded = sae.decode(encoded)
        decoded_shape = _shape(decoded)
        compatible = sae_shapes_are_compatible(activation_shape, decoded_shape)
        if not compatible:
            raise ValueError(
                "decoded SAE output is not shape-compatible with hook activation: "
                f"decoded={decoded_shape}, activation={activation_shape}"
            )

        result = SAECompatibilityResult(
            model_name=model_name,
            hook_point=hook_point,
            sae_release=sae_release,
            sae_id=sae_id,
            activation_shape=activation_shape,
            encoded_shape=encoded_shape,
            decoded_shape=decoded_shape,
            d_model=d_model,
            d_sae=d_sae,
            compatible=True,
            status="ok",
        )
    except Exception as exc:
        result = SAECompatibilityResult(
            model_name=model_name,
            hook_point=hook_point,
            sae_release=sae_release,
            sae_id=sae_id,
            activation_shape=activation_shape,
            encoded_shape=encoded_shape,
            decoded_shape=decoded_shape,
            d_model=d_model,
            d_sae=d_sae,
            compatible=False,
            status="error",
            error=str(exc),
        )

    if out is not None:
        write_config(result.model_dump(), out)
    return result
