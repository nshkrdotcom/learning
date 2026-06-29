from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from construct_mismatch.datasets import CONSTRUCTS


def run(command: list[str], cwd: Path) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-pairs-per-axis", type=int, default=2)
    parser.add_argument("--max-steering-examples", type=int, default=24)
    args = parser.parse_args()

    root = args.output_root
    py = sys.executable
    run([py, "scripts/inspect_tokenization.py", "--model", args.model, "--device", args.device], root)
    run([py, "scripts/build_datasets.py", "--model", args.model, "--device", args.device], root)
    run([py, "scripts/check_behavior.py", "--model", args.model, "--device", args.device], root)
    for construct in CONSTRUCTS:
        run(
            [
                py,
                "scripts/run_direction_experiment.py",
                "--model",
                args.model,
                "--device",
                args.device,
                "--construct",
                construct,
                "--max-steering-examples",
                str(args.max_steering_examples),
            ],
            root,
        )
        run(
            [
                py,
                "scripts/run_probe_experiment.py",
                "--model",
                args.model,
                "--device",
                args.device,
                "--construct",
                construct,
            ],
            root,
        )
        run(
            [
                py,
                "scripts/run_patching_experiment.py",
                "--model",
                args.model,
                "--device",
                args.device,
                "--construct",
                construct,
                "--max-pairs-per-axis",
                str(args.max_pairs_per_axis),
            ],
            root,
        )
    run([py, "scripts/make_figures.py"], root)


if __name__ == "__main__":
    main()
