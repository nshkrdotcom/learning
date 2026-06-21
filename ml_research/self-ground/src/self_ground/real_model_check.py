from __future__ import annotations

from pathlib import Path
from typing import Any

from self_ground.io import write_config
from self_ground.model import TransformerLensModelAdapter

DEFAULT_CHECK_TEXTS = [
    "The dog is friendly.",
    "The dog is not friendly.",
    "The dog is often friendly.",
    "The dog isn't friendly.",
]


def check_real_model(
    *,
    model_name: str = "EleutherAI/pythia-70m",
    hook_point: str = "blocks.2.hook_resid_post",
    out: str | Path = "runs/check_real_model.json",
    device: str | None = "cpu",
    texts: list[str] | None = None,
) -> dict[str, Any]:
    check_texts = texts or DEFAULT_CHECK_TEXTS
    try:
        adapter = TransformerLensModelAdapter(model_name=model_name, device=device)
        activations = adapter.get_activations(check_texts, hook_point=hook_point)
    except Exception as exc:
        raise RuntimeError(
            "real model check failed while loading the TransformerLens model or "
            f"capturing hook point {hook_point!r} from model {model_name!r}: {exc}"
        ) from exc

    artifact = {
        "model": model_name,
        "hook_point": hook_point,
        "texts": check_texts,
        "activation_shape": list(activations.shape),
        "dtype": str(activations.dtype),
        "device": str(activations.device),
        "status": "ok",
    }
    write_config(artifact, out)
    return artifact
