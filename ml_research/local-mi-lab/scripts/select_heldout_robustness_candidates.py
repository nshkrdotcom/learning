from __future__ import annotations

import argparse

from local_mi_lab.robustness_candidates import (
    select_heldout_candidate_set,
    write_heldout_candidate_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiseed", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-missing-required", action="store_true")
    args = parser.parse_args()

    candidates = select_heldout_candidate_set(
        args.multiseed,
        allow_missing_required=args.allow_missing_required,
    )
    paths = write_heldout_candidate_artifacts(
        candidates,
        args.output,
        source_multiseed=args.multiseed,
    )
    print(paths["csv"])


if __name__ == "__main__":
    main()
