"""Compute BPW.

Usage:
    python bpw.py input.csv
"""
import argparse
import json
import logging
import os
import os.path as osp
import sys

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm import tqdm

os.environ["HF_HOME"] = f"{osp.dirname(__file__)}/../tmp_saves/hg_cache"
sys.path.append(f"{osp.dirname(osp.abspath(__file__))}/../src")

import codec


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bitstring to stegotext based on cover text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ###############
    #             #
    #    I / O    #
    #             #
    ###############
    parser.add_argument("input", type=str, help="Input .csv file.")
    parser.add_argument(
        "--data-col",
        type=str,
        default="stegotext",
    )
    parser.add_argument(
        "--bit-col",
        type=str,
        default="enc_bits",
    )
    parser.add_argument(
        "--used-bits-col",
        type=str,
        default="used_bits",
        help="Column name of the length of used bits.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=None,
        help="Output .json file if needed.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite output file.",
    )
    ####################
    #                  #
    #    validating    #
    #                  #
    ####################
    args = parser.parse_args()
    assert osp.exists(args.input), f"{args.input} does not exist."
    if not args.force:
        assert args.output is None or not osp.exists(
            args.output
        ), f"{args.output} already exists. Use --force to overwrite."
    # different mode check
    return args


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    ###################
    #                 #
    #    load data    #
    #                 #
    ###################
    logging.info(f"Loading input data: {args.input}.")
    df = pd.read_csv(args.input)
    if args.used_bits_col not in df.columns:
        logging.warning(
            f"{args.used_bits_col} not found in input data. Using the length of bit column `{args.bit_col}` instead."
        )
    bpws = []
    for row in tqdm(df.itertuples(), total=len(df)):
        text = getattr(row, args.data_col)
        word_len = len(word_tokenize(text))
        if hasattr(row, args.used_bits_col):
            # if the `used_bits` column exists, use it
            bits_len = int(getattr(row, args.used_bits_col))
        else:
            # otherwise, use the length of the bitstream
            bits_base64 = getattr(row, args.bit_col)
            bits = codec.base642bits(bits_base64)
            bits_len = len(bits)
        bpws.append(bits_len / word_len)

    print("Avg. BPW:", np.mean(bpws))
    if args.output is not None:
        logging.info(f"Saving output to {args.output}.")
        with open(args.output, "w") as f:
            json.dump(
                {
                    "avg_bpw": np.mean(bpws),
                },
                f,
            )
