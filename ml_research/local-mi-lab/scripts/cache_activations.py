from __future__ import annotations

import argparse
import json

from local_mi_lab.activations import cache_selected_resid_post
from local_mi_lab.config import load_config
from local_mi_lab.models import load_hooked_transformer
from local_mi_lab.paths import resolve_repo_path
from local_mi_lab.prompts import read_prompts_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = resolve_repo_path(args.run)
    records = read_prompts_csv(run_dir / "prompts.csv")
    model = load_hooked_transformer(config)
    model.eval()
    manifest = cache_selected_resid_post(model, records, config, run_dir)
    summary = {
        "model": manifest["model"],
        "layers": manifest["layers"],
        "n_examples": manifest["n_examples"],
        "files": manifest["files"],
    }
    (run_dir / "activation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(run_dir / "activations" / "manifest.json")


if __name__ == "__main__":
    main()
