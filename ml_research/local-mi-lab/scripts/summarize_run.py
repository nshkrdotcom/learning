from __future__ import annotations

import argparse

from local_mi_lab.report import write_run_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    print(write_run_summary(args.run))


if __name__ == "__main__":
    main()
