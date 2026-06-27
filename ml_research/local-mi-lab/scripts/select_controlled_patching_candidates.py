from __future__ import annotations

import argparse

from local_mi_lab.candidate_selection import (
    select_controlled_patching_candidates,
    write_candidate_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--top-k-raw", type=int, default=5)
    parser.add_argument("--top-k-control", type=int, default=5)
    parser.add_argument("--top-k-gap", type=int, default=5)
    parser.add_argument("--n-random", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    candidates = select_controlled_patching_candidates(
        args.run,
        top_k_raw=args.top_k_raw,
        top_k_control=args.top_k_control,
        top_k_gap=args.top_k_gap,
        n_random=args.n_random,
        seed=args.seed,
    )
    paths = write_candidate_artifacts(
        args.run,
        candidates,
        top_k_raw=args.top_k_raw,
        top_k_control=args.top_k_control,
        top_k_gap=args.top_k_gap,
        n_random=args.n_random,
        seed=args.seed,
    )
    print(paths["csv"])


if __name__ == "__main__":
    main()
