import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    shards = sorted(data_root.glob("*.npy"))
    if not shards:
        raise SystemExit(f"No .npy shards found in {data_root}")
    for path in shards:
        x = np.load(path)
        print(path, x.shape, x.dtype, int(x.min()), int(x.max()))
        if x.max() >= 2**16:
            raise SystemExit(f"Token id too large for uint16 shard: {path}")


if __name__ == "__main__":
    main()

