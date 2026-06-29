from __future__ import annotations

import argparse
import csv
from pathlib import Path

from construct_mismatch.datasets import artifact_path
from construct_mismatch.model import decode_tokens, encode_text, load_model
from construct_mismatch.tokenization import inspect_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=artifact_path() / "tokenization" / "gpt2_small_target_tokens.csv",
    )
    args = parser.parse_args()

    model = load_model(args.model, args.device)
    rows = inspect_candidates(
        encode_text=lambda text: encode_text(model, text),
        decode_tokens=lambda token_ids: decode_tokens(model, token_ids),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "construct",
                "raw_string",
                "token_ids",
                "decoded_text",
                "n_tokens",
                "is_single_token",
                "usable_as_target",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "construct": row.construct,
                    "raw_string": row.raw_string,
                    "token_ids": " ".join(str(token_id) for token_id in row.token_ids),
                    "decoded_text": row.decoded_text,
                    "n_tokens": row.n_tokens,
                    "is_single_token": row.is_single_token,
                    "usable_as_target": row.usable_as_target,
                }
            )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
