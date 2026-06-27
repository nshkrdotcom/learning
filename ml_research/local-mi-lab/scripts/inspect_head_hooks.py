from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from local_mi_lab.config import load_config
from local_mi_lab.head_hooks import inspect_head_hooks, resolve_head_patch_site_from_metadata
from local_mi_lab.models import load_hooked_transformer
from local_mi_lab.paths import make_run_dir
from local_mi_lab.resources import collect_resource_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--layers", default="0")
    args = parser.parse_args()

    config = load_config(args.config)
    layers = [int(layer) for layer in args.layers.split(",") if layer.strip()]
    run_dir = make_run_dir(config["outputs"]["run_root"], "head_hook_inspection")
    model = load_hooked_transformer(config)
    model.eval()
    rows = inspect_head_hooks(model, args.prompt, layers)
    resolutions = [
        resolve_head_patch_site_from_metadata(rows, layer, preferred="hook_z").__dict__
        for layer in layers
    ]
    report = {
        "config": args.config,
        "model": config["model"]["name"],
        "prompt": args.prompt,
        "layers": layers,
        "hook_inspections": rows,
        "head_patch_site_resolutions": resolutions,
        "resources": collect_resource_snapshot("."),
    }
    write_inspection(report, run_dir)
    print(run_dir)


def write_inspection(report: dict[str, Any], run_dir: Path) -> None:
    (run_dir / "head_hook_inspection.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Head Hook Inspection",
        "",
        f"Model: `{report['model']}`",
        f"Prompt: `{report['prompt']}`",
        f"Layers: `{report['layers']}`",
        "",
        "## Patch Site Resolution",
        "",
    ]
    for site in report["head_patch_site_resolutions"]:
        lines.append(
            "- "
            f"`{site['hook_name'] or 'unsupported'}`: "
            f"head_specific={site['head_specific_possible']}, "
            f"scope=`{site['actual_patch_scope']}`"
        )
    lines.extend(["", "## Inspected Hooks", ""])
    for row in report["hook_inspections"]:
        lines.append(
            "- "
            f"`{row['hook_name']}`: captured={row['exists_or_capturable']}, "
            f"shape={row['tensor_shape']}, "
            f"head_axis={row['head_dimension_axis']}, "
            f"seq_axis={row['seq_axis']}, "
            f"notes={row['notes']}"
        )
    (run_dir / "head_hook_inspection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
