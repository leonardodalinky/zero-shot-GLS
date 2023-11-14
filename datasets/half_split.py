"""Script to split the dataset into two halves."""
import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser("Script to split the dataset into two halves.")
    parser.add_argument("input", type=Path, help="Path to the input .csv dataset.")
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Whether to shuffle the dataset before splitting.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")

    args = parser.parse_args()
    assert args.input.exists(), f"Input file {args.input} does not exist."

    return args


if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(args.input)

    if not args.no_shuffle:
        df = df.sample(frac=1.0, random_state=args.seed, ignore_index=True)
    # halve the dataset
    df_1 = df.iloc[: len(df) // 2]
    df_2 = df.iloc[len(df) // 2 :]

    new_path_1 = args.input.parent / f"{args.input.stem}.part1.csv"
    new_path_2 = args.input.parent / f"{args.input.stem}.part2.csv"

    df_1.to_csv(new_path_1, index=False)
    df_2.to_csv(new_path_2, index=False)
