from __future__ import annotations

import argparse

from local_mi_lab.config import load_config, max_examples_for_initial_run
from local_mi_lab.prompts import generate_induction_prompts, write_prompt_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    if config["task"]["name"] != "induction":
        raise ValueError("build_toy_prompts.py currently builds the induction prompt dataset")
    records = generate_induction_prompts(
        n_examples=max_examples_for_initial_run(config),
        seed=int(config["experiment"].get("seed", 0)),
    )
    output_dir = write_prompt_dataset(records, "data", "induction_prompts_v0")
    print(output_dir)


if __name__ == "__main__":
    main()
