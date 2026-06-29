from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def verify_data_root(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    shards = sorted(data_root.glob("*.npy"))
    if not shards:
        raise SystemExit(f"No .npy shards found in {data_root}")
    for path in shards:
        x = np.load(path)
        if x.size == 0:
            raise SystemExit(f"Empty shard: {path}")
        min_token = int(x.min())
        max_token = int(x.max())
        print(path, x.shape, x.dtype, min_token, max_token)
        if min_token < 0:
            raise SystemExit(f"Negative token id found in shard: {path}")
        if max_token >= 2**16:
            raise SystemExit(f"Token id too large for uint16 shard: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    args = parser.parse_args()
    verify_data_root(args)


if __name__ == "__main__":
    main()

