from __future__ import annotations

import argparse

from local_mi_lab.config import load_config
from local_mi_lab.heldout_prompts import generate_heldout_induction_prompts
from local_mi_lab.prompts import write_prompt_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    if config["task"]["name"] != "induction_heldout":
        raise ValueError("build_heldout_induction_prompts.py requires task.name=induction_heldout")
    seed = int(config["experiment"].get("seed", 0))
    records = generate_heldout_induction_prompts(
        n_examples_per_family=int(config["task"]["n_examples_initial_per_family"]),
        families=list(config["task"]["families"]),
        seed=seed,
    )
    dataset_name = f"induction_heldout_seed{seed}"
    output_dir = write_prompt_dataset(records, "data", dataset_name)
    print(output_dir)


if __name__ == "__main__":
    main()
