"""
Compute average PPL.
"""
import argparse
import json
import os
import os.path as osp

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        "Compute PPL based on trained GPT-2 model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ###############
    #             #
    #    I / O    #
    #             #
    ###############
    parser.add_argument("input", type=str, help="Path to the input .csv data file.")
    parser.add_argument(
        "--ppl-col",
        type=str,
        default="ppl",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output .json file.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite output file.",
    )
    ####################
    #                  #
    #    Validating    #
    #                  #
    ####################
    args = parser.parse_args()
    assert osp.exists(args.input), f"Input file {args.input} does not exist."
    assert args.force or not osp.exists(
        args.output
    ), f"{args.output} already exists. Use --force to overwrite."

    return args


if __name__ == "__main__":
    args = parse_args()
    ###################
    #                 #
    #    load data    #
    #                 #
    ###################
    df = pd.read_csv(args.input)
    assert args.ppl_col in df.columns, f"Data Column '{args.ppl_col}' not in {args.input}."

    ppl_mean = np.mean(df[args.ppl_col].values)
    print(f"Average PPL: {ppl_mean:.4f}")

    print("Output to:", args.output)
    with open(args.output, "w") as fp:
        json.dump(
            {
                "ppl_mean": ppl_mean,
            },
            fp,
        )
