from __future__ import annotations

import argparse

from local_mi_lab.config import load_config, max_examples_for_initial_run
from local_mi_lab.prompts import (
    generate_induction_control_prompts,
    generate_induction_prompts,
    write_prompt_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    task_name = config["task"]["name"]
    seed = int(config["experiment"].get("seed", 0))
    if task_name == "induction":
        records = generate_induction_prompts(
            n_examples=max_examples_for_initial_run(config),
            seed=seed,
        )
        dataset_name = "induction_prompts_v0"
    elif task_name == "induction_controls":
        records = generate_induction_control_prompts(
            n_examples_per_family=int(config["task"]["n_examples_initial_per_family"]),
            families=list(config["task"]["families"]),
            seed=seed,
        )
        dataset_name = "induction_controls_v0"
    else:
        raise ValueError(f"Unsupported prompt-building task: {task_name}")
    output_dir = write_prompt_dataset(records, "data", dataset_name)
    print(output_dir)


if __name__ == "__main__":
    main()
