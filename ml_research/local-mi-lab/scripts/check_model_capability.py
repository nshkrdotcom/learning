from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import torch

from local_mi_lab.activations import resid_post_hook_name
from local_mi_lab.config import load_config, max_examples_for_initial_run, selected_layers
from local_mi_lab.models import load_hooked_transformer, n_layers
from local_mi_lab.paths import make_run_dir
from local_mi_lab.resources import (
    collect_resource_snapshot,
    safe_initial_batch_size,
    transformer_lens_import_status,
)
from local_mi_lab.tokens import encode_text, token_id_for_single_token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = make_run_dir(config["outputs"]["run_root"], "capability_check")
    report = run_capability_check(config, args.config)
    write_reports(report, run_dir)
    print(run_dir)
    if report["status"] != "ok":
        raise SystemExit(1)


def run_capability_check(config: dict[str, Any], config_path: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "started",
        "config": config_path,
        "model": config["model"]["name"],
        "checks": {},
        "blocker": None,
    }
    try:
        resources = collect_resource_snapshot(".")
        report["checks"]["resources"] = resources
        free_vram = resources["cuda"].get("free_vram_gb") if resources["cuda"]["available"] else None
        report["checks"]["safe_initial_batch_size"] = safe_initial_batch_size(
            max_examples=max_examples_for_initial_run(config),
            max_gpu_vram_fraction=float(config["resources"]["max_gpu_vram_fraction"]),
            free_vram_gb=free_vram,
        )

        tl_status = transformer_lens_import_status()
        report["checks"]["transformer_lens_import"] = tl_status
        if not tl_status["ok"]:
            raise RuntimeError(f"TransformerLens import failed: {tl_status['error']}")

        model = load_hooked_transformer(config)
        model.eval()
        report["checks"]["model_load"] = {
            "ok": True,
            "n_layers": n_layers(model),
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype),
        }

        token_ids = encode_text(model.tokenizer, " A B C")
        expected_id = token_id_for_single_token(model.tokenizer, " D")
        report["checks"]["tokenizer_behavior"] = {
            "ok": True,
            "sample_text": " A B C",
            "sample_token_ids": token_ids,
            "expected_token": " D",
            "expected_token_id": expected_id,
        }

        with torch.inference_mode():
            one_tokens = model.to_tokens("A")
            one_logits = model(one_tokens)
            batch_tokens = model.to_tokens(["A B C", "D E F"])
            batch_logits = model(batch_tokens)
        report["checks"]["one_token_forward_pass"] = {
            "ok": True,
            "input_shape": list(one_tokens.shape),
            "logits_shape": list(one_logits.shape),
        }
        report["checks"]["small_batch_forward_pass"] = {
            "ok": True,
            "input_shape": list(batch_tokens.shape),
            "logits_shape": list(batch_logits.shape),
        }

        layer = selected_layers(config, n_layers(model))[0]
        hook_name = resid_post_hook_name(layer)
        with torch.inference_mode():
            _, cache = model.run_with_cache(one_tokens, names_filter=[hook_name])
        report["checks"]["basic_activation_cache"] = {
            "ok": True,
            "hook_name": hook_name,
            "shape": list(cache[hook_name].shape),
        }
        report["status"] = "ok"
    except Exception as exc:  # pragma: no cover - exercised only on local dependency failures
        report["status"] = "failed"
        report["blocker"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    return report


def write_reports(report: dict[str, Any], run_dir: Path) -> None:
    json_path = run_dir / "capability_report.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Capability Report",
        "",
        f"Status: `{report['status']}`",
        f"Model: `{report['model']}`",
        "",
        "## Checks",
        "",
    ]
    for name, value in report["checks"].items():
        lines.append(f"- `{name}`: present")
        if name == "resources":
            cuda = value["cuda"]
            lines.append(f"  - CUDA available: {cuda.get('available')}")
            if cuda.get("available"):
                lines.append(f"  - GPU: {cuda.get('gpu_name')}")
                lines.append(f"  - Free VRAM GB: {cuda.get('free_vram_gb'):.2f}")
            lines.append(f"  - Available RAM GB: {value.get('system_ram_available_gb'):.2f}")
            lines.append(f"  - Free disk GB: {value.get('disk_free_gb'):.2f}")
    if report["blocker"]:
        lines.extend(
            [
                "",
                "## Blocker",
                "",
                f"Type: `{report['blocker']['type']}`",
                "",
                report["blocker"]["message"],
            ]
        )
    (run_dir / "capability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
