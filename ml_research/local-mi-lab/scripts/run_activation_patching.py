from __future__ import annotations

import argparse

from local_mi_lab.config import load_config
from local_mi_lab.models import load_hooked_transformer
from local_mi_lab.patching import run_resid_post_patching
from local_mi_lab.paths import make_run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--clean-prompt")
    parser.add_argument("--corrupt-prompt")
    parser.add_argument("--target-token")
    parser.add_argument("--full-sweep", action="store_true")
    parser.add_argument("--component", choices=["resid_post", "attn_out"])
    parser.add_argument("--allow-length-mismatch", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    patching_config = (config.get("task") or {}).get("patching", {})
    clean_prompt = args.clean_prompt or patching_config.get("clean_prompt")
    corrupt_prompt = args.corrupt_prompt or patching_config.get("corrupt_prompt")
    target_token = args.target_token or patching_config.get("target_token")
    component = args.component or patching_config.get("component", "resid_post")
    if not clean_prompt or not corrupt_prompt or not target_token:
        raise ValueError(
            "Activation patching requires explicit clean prompt, corrupt prompt, and target token "
            "from CLI args or config task.patching."
        )

    run_dir = make_run_dir(config["outputs"]["run_root"], config["experiment"]["name"])
    model = load_hooked_transformer(config)
    model.eval()
    run_resid_post_patching(
        model,
        clean_prompt=clean_prompt,
        corrupt_prompt=corrupt_prompt,
        target_token=target_token,
        config=config,
        output_dir=run_dir,
        full_sweep=args.full_sweep,
        component=component,
        allow_length_mismatch=args.allow_length_mismatch,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
