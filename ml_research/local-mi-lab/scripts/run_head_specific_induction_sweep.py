from __future__ import annotations

import argparse

from local_mi_lab.config import load_config
from local_mi_lab.head_sweep import run_head_specific_induction_sweep
from local_mi_lab.models import load_hooked_transformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-run")
    parser.add_argument("--layers")
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--examples-per-family", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    model = load_hooked_transformer(config)
    model.eval()
    layer_override = "all" if args.all_layers else args.layers
    run_dir = run_head_specific_induction_sweep(
        model,
        config,
        args.source_run,
        output_run=args.output_run,
        layers_override=layer_override,
        examples_per_family_override=args.examples_per_family,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
