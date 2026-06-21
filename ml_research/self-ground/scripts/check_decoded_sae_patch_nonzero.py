from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from self_ground.baselines import select_top_features
from self_ground.io import write_config
from self_ground.model import TransformerLensModelAdapter
from self_ground.sae import SAELensAdapter
from self_ground.sae_compat import verify_sae_compatibility
from self_ground.sae_interventions import (
    decoded_sae_patch_with_telemetry_from_activation,
    modify_sae_features,
    parse_sae_feature_id,
    run_sae_decoded_intervention_logits_with_telemetry,
)


def _tensor(value: Any, *, like: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def _selected_values(
    encoded: torch.Tensor,
    feature_indices: list[int],
    token_position: int,
) -> torch.Tensor:
    position = token_position if token_position >= 0 else encoded.shape[1] + token_position
    if encoded.ndim == 2:
        return encoded[:, feature_indices]
    return encoded[:, position, feature_indices]


def patch_check_notes(
    *,
    feature_delta_l1: float,
    decoded_delta_norm: float,
    max_abs_logit_delta: float,
) -> list[str]:
    notes: list[str] = []
    if feature_delta_l1 == 0.0:
        notes.append("selected_features_inactive_or_no_feature_change")
    if feature_delta_l1 > 0.0 and decoded_delta_norm == 0.0:
        notes.append("possible_sae_decode_issue")
    if decoded_delta_norm > 0.0 and max_abs_logit_delta == 0.0:
        notes.append("possible_patch_or_hook_issue")
    if max_abs_logit_delta > 0.0:
        notes.append("decoded_patch_changes_logits")
    return notes


def run_patch_nonzero_check(
    *,
    out_dir: str | Path,
    ranking_dir: str | Path,
    model_name: str,
    hook_point: str,
    sae_release: str,
    sae_id: str,
    top_k_features: int,
    operation: str,
    patch_mode: str,
    prompt: str,
    device: str,
    factor: float,
    target_token: str | None = None,
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    config = {
        "ranking_dir": str(ranking_dir),
        "model_name": model_name,
        "hook_point": hook_point,
        "sae_release": sae_release,
        "sae_id": sae_id,
        "top_k_features": top_k_features,
        "operation": operation,
        "patch_mode": patch_mode,
        "prompt": prompt,
        "device": device,
        "factor": factor,
        "target_token": target_token,
    }
    write_config(config, out_path / "config.json")
    try:
        feature_ids = select_top_features(Path(ranking_dir), top_k=top_k_features)
        model_adapter = TransformerLensModelAdapter(model_name=model_name, device=device)
        sae_adapter = SAELensAdapter.from_pretrained(
            release=sae_release,
            sae_id=sae_id,
            device=device,
        )
        compatibility = verify_sae_compatibility(
            model_name=model_name,
            hook_point=hook_point,
            sae_release=sae_release,
            sae_id=sae_id,
            device=device,
            model_adapter=model_adapter,
            sae_adapter=sae_adapter,
        )
        activation = model_adapter.get_activations([prompt], hook_point)
        encoded = _tensor(sae_adapter.encode(activation).values, like=activation)
        feature_indices = [parse_sae_feature_id(feature_id) for feature_id in feature_ids]
        modified = modify_sae_features(
            encoded,
            feature_indices,
            operation=operation,  # type: ignore[arg-type]
            factor=factor,
            token_position=-1,
        )
        original_values = _selected_values(encoded, feature_indices, -1)
        modified_values = _selected_values(modified, feature_indices, -1)
        feature_delta_l1 = float(
            (modified_values - original_values).abs().sum().detach().cpu().item()
        )
        _, telemetry = decoded_sae_patch_with_telemetry_from_activation(
            activation=activation,
            sae_adapter=sae_adapter,
            feature_ids=feature_ids,
            operation=operation,  # type: ignore[arg-type]
            factor=factor,
            token_position=-1,
            patch_mode=patch_mode,  # type: ignore[arg-type]
        )
        baseline_logits = model_adapter.logits_for_texts([prompt])
        patched_logits, _ = run_sae_decoded_intervention_logits_with_telemetry(
            model_adapter,
            sae_adapter,
            [prompt],
            hook_point,
            feature_ids,
            operation=operation,  # type: ignore[arg-type]
            factor=factor,
            token_position=-1,
            patch_mode=patch_mode,  # type: ignore[arg-type]
        )
        logit_delta = patched_logits - baseline_logits
        max_abs_logit_delta = float(logit_delta.abs().max().detach().cpu().item())
        target_logit_delta = None
        if target_token:
            target_id = model_adapter.token_ids_for_strings([target_token])[0]
            target_logit_delta = float(logit_delta[0, -1, target_id].detach().cpu().item())
        notes = patch_check_notes(
            feature_delta_l1=feature_delta_l1,
            decoded_delta_norm=telemetry.decoded_delta_norm_mean,
            max_abs_logit_delta=max_abs_logit_delta,
        )
        result = {
            "status": "ok",
            "compatible": compatibility.compatible,
            "selected_feature_ids": feature_ids,
            "original_feature_values": original_values.detach().cpu().reshape(-1).tolist(),
            "modified_feature_values": modified_values.detach().cpu().reshape(-1).tolist(),
            "feature_delta_l1": feature_delta_l1,
            "decoded_delta_norm": telemetry.decoded_delta_norm_mean,
            "residual_norm": telemetry.activation_norm_mean,
            "relative_decoded_delta_norm": telemetry.decoded_delta_norm_ratio,
            "patched_logits_changed": max_abs_logit_delta > 0.0,
            "max_abs_logit_delta": max_abs_logit_delta,
            "target_token": target_token,
            "target_logit_delta": target_logit_delta,
            "notes": notes,
        }
    except Exception as exc:
        result = {
            "status": "blocked",
            "compatible": False,
            "selected_feature_ids": [],
            "original_feature_values": [],
            "modified_feature_values": [],
            "feature_delta_l1": 0.0,
            "decoded_delta_norm": 0.0,
            "residual_norm": 0.0,
            "relative_decoded_delta_norm": 0.0,
            "patched_logits_changed": False,
            "max_abs_logit_delta": 0.0,
            "target_token": target_token,
            "target_logit_delta": None,
            "blocker": {
                "exception_class": type(exc).__name__,
                "exception_message": str(exc),
            },
            "notes": ["patch_check_blocked"],
        }
    write_config(result, out_path / "patch_check.json")
    readme = f"""# Decoded SAE Patch Nonzero Check

- status: `{result["status"]}`
- selected features: `{result["selected_feature_ids"]}`
- feature delta L1: `{result["feature_delta_l1"]}`
- decoded delta norm: `{result["decoded_delta_norm"]}`
- max abs logit delta: `{result["max_abs_logit_delta"]}`

This is a targeted sanity check for the real decoded SAE patch path. It is not a
Phase 3 evidence run.
"""
    (out_path / "README.md").write_text(readme, encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether a decoded SAE patch is nonzero.")
    parser.add_argument("--ranking-dir", required=True)
    parser.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--hook-point", default="blocks.2.hook_resid_post")
    parser.add_argument("--sae-release", default="pythia-70m-deduped-res-sm")
    parser.add_argument("--sae-id", default="blocks.2.hook_resid_post")
    parser.add_argument("--top-k-features", type=int, default=2)
    parser.add_argument("--operation", default="ablate", choices=["ablate", "amplify"])
    parser.add_argument("--patch-mode", default="delta", choices=["replace", "delta"])
    parser.add_argument("--prompt", default="The movie was not")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    parser.add_argument("--factor", type=float, default=2.0)
    parser.add_argument("--target-token")
    args = parser.parse_args()
    factor = 1.0 if args.operation == "ablate" else args.factor
    result = run_patch_nonzero_check(
        out_dir=args.out,
        ranking_dir=args.ranking_dir,
        model_name=args.model,
        hook_point=args.hook_point,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        top_k_features=args.top_k_features,
        operation=args.operation,
        patch_mode=args.patch_mode,
        prompt=args.prompt,
        device=args.device,
        factor=factor,
        target_token=args.target_token,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
