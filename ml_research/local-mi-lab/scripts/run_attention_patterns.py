from __future__ import annotations

import argparse

from local_mi_lab.attention import compute_attention_patterns
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
    compute_attention_patterns(model, records, config, run_dir)
    print(run_dir / "attention_summary.json")


if __name__ == "__main__":
    main()
