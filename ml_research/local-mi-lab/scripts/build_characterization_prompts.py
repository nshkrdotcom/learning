from __future__ import annotations

import argparse

from local_mi_lab.characterization_prompts import generate_characterization_prompts
from local_mi_lab.config import load_config
from local_mi_lab.prompts import write_prompt_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["experiment"].get("seed", 20))
    records = generate_characterization_prompts(
        n_examples_per_family=int(config["task"]["n_examples_initial_per_family"]),
        families=list(config["task"]["families"]),
        seed=seed,
    )
    dataset = f"candidate_characterization_seed{seed}"
    root = write_prompt_dataset(records, "data", dataset)
    print(root)


if __name__ == "__main__":
    main()
