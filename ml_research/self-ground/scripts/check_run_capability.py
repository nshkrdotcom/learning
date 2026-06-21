from __future__ import annotations

import argparse
import importlib
import json
import platform
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from self_ground.io import write_config


def _import_status(
    package: str,
    *,
    importer: Callable[[str], ModuleType],
) -> tuple[bool, dict[str, str] | None, ModuleType | None]:
    try:
        module = importer(package)
    except Exception as exc:
        return (
            False,
            {
                "type": "import_failed",
                "package": package,
                "exception_class": type(exc).__name__,
                "exception_message": str(exc),
            },
            None,
        )
    return True, None, module


def collect_run_capability(
    *,
    out_dir: str | Path,
    model: str,
    sae_release: str,
    sae_id: str,
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    config = {
        "model_target": model,
        "sae_release": sae_release,
        "sae_id": sae_id,
    }
    write_config(config, out_path / "config.json")

    blockers: list[dict[str, str]] = []
    torch_ok, torch_blocker, torch_module = _import_status("torch", importer=importer)
    if torch_blocker:
        blockers.append(torch_blocker)
    tl_ok, tl_blocker, _ = _import_status("transformer_lens", importer=importer)
    if tl_blocker:
        blockers.append(tl_blocker)
    sae_ok, sae_blocker, _ = _import_status("sae_lens", importer=importer)
    if sae_blocker:
        blockers.append(sae_blocker)

    cuda_available = False
    cuda_device_count = 0
    cuda_device_names: list[str] = []
    torch_version = None
    if torch_module is not None:
        torch_version = str(getattr(torch_module, "__version__", "unknown"))
        cuda = getattr(torch_module, "cuda", None)
        if cuda is not None:
            try:
                cuda_available = bool(cuda.is_available())
                cuda_device_count = int(cuda.device_count()) if cuda_available else 0
                cuda_device_names = [
                    str(cuda.get_device_name(index)) for index in range(cuda_device_count)
                ]
            except Exception as exc:
                blockers.append(
                    {
                        "type": "cuda_query_failed",
                        "exception_class": type(exc).__name__,
                        "exception_message": str(exc),
                    }
                )
    if torch_ok and not cuda_available:
        blockers.append(
            {
                "type": "cuda_unavailable",
                "exception_class": "",
                "exception_message": "torch.cuda.is_available() returned False",
            }
        )

    can_attempt = torch_ok and tl_ok and sae_ok and cuda_available
    capability = {
        "python_version": platform.python_version(),
        "torch_available": torch_ok,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_names": cuda_device_names,
        "transformer_lens_importable": tl_ok,
        "sae_lens_importable": sae_ok,
        "model_target": model,
        "sae_release": sae_release,
        "sae_id": sae_id,
        "can_attempt_e002_gpu": can_attempt,
        "blockers": blockers if not can_attempt else [],
    }
    write_config(capability, out_path / "capability.json")
    readme = f"""# SELF-GROUND Run Capability Check

- model: `{model}`
- SAE release: `{sae_release}`
- SAE id: `{sae_id}`
- CUDA available: `{cuda_available}`
- can attempt E002 GPU: `{can_attempt}`

Blockers are recorded in `capability.json`. This check imports installed
packages and queries CUDA; it does not load the model or SAE weights.
"""
    (out_path / "README.md").write_text(readme, encoding="utf-8")
    return capability


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether E002 GPU run can be attempted.")
    parser.add_argument("--out", default="runs/capability_check")
    parser.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--sae-release", default="pythia-70m-deduped-res-sm")
    parser.add_argument("--sae-id", default="blocks.2.hook_resid_post")
    args = parser.parse_args()
    capability = collect_run_capability(
        out_dir=args.out,
        model=args.model,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
    )
    print(json.dumps(capability, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
