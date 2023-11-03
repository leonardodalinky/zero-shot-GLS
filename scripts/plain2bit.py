"""Plain-text to bitstring.

Usage:
    TODO
"""
import argparse
import csv
import logging
import os
import os.path as osp
import sys
from typing import Any

from tqdm import tqdm
from transformers import GPT2Model, GPT2Tokenizer

sys.path.append(f"{osp.dirname(osp.abspath(__file__))}/../src")

import codec


def parse_args():
    parser = argparse.ArgumentParser(description="Plain-text to bitstring.")
    parser.add_argument("input", type=str, help="Input .csv file.")
    parser.add_argument(
        "-s",
        "--src-col",
        type=str,
        default="plaintext",
        help="Column name of source plain-text. [Default: plaintext]",
    )
    parser.add_argument(
        "-d",
        "--dst-col",
        type=str,
        default="bits",
        help="Column name of destination bitstring. [Default: enc_bits]",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output .csv file.",
    )
    parser.add_argument(
        "-n" "--n-rows",
        type=int,
        help="Number of rows to process. [Default: all]",
    )
    parser.add_argument(
        "--skip",
        type=int,
        help="Skip first N rows.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite output file.",
    )
    ##########################
    #                        #
    #    Encoding options    #
    #                        #
    ##########################
    parser.add_argument(
        "--size-bits",
        type=int,
        default=10,
        help="Number of size bits. [Default: 4]",
    )
    parser.add_argument(
        "--ef-rounds",
        type=int,
        default=4,
        help="Number of EF rounds. [Default: 4]",
    )
    ####################
    #                  #
    #    validating    #
    #                  #
    ####################
    assert osp.exists(args.input), f"{args.input} does not exist."
    assert args.force or not osp.exists(
        args.output
    ), f"{args.output} already exists. Use --force to overwrite."
    assert args.size_bits > 0, f"--size-bits must be greater than 0."
    assert args.ef_rounds >= 0, f"--ef-rounds must be greater than or equal to 0."

    return parser.parse_args()


def encode(
    model: GPT2Model,
    tokenizer: GPT2Tokenizer,
    plaintext: str,
    size_bits: int,
    ef_rounds: int,
) -> str:
    """Encode plain-text to bitstring.

    Args:
        model (GPT2Model): GPT-2 model.
        tokenizer (GPT2Tokenizer): GPT-2 tokenizer.
        plaintext (str): Plain-text.
        size_bits (int): Number of size bits.
        ef_rounds (int): Number of EF rounds.

    Returns:
        str: Bitstring encoded in Base64 format.
    """
    ...


if __name__ == "__main__":
    args = parse_args()
    ###################
    #                 #
    #    load data    #
    #                 #
    ###################
    logging.info(f"Loading input data: {args.input}.")
    with open(args.input, "r") as fp:
        dialect = csv.Sniffer().sniff(fp.read(1024))
        fp.seek(0)
        reader = csv.DictReader(fp, dialect=dialect)
        input_fieldnames = list(reader.fieldnames)
        input_data: list[dict[str, Any]] = list(reader)
    assert args.src_col in reader.fieldnames, f"Src Column {args.src_col} not in {args.input}."
    assert (
        args.dst_col not in reader.fieldnames
    ), f"Dst column {args.dst_col} already in {args.input}."
    if args.skip is not None:
        assert args.skip > 0, f"--skip must be greater than 0."
        logging.info(f"Skipping first {args.skip} rows.")
        start_idx = args.skip
    else:
        start_idx = 0
    if args.n_rows is not None:
        assert args.n_rows > 0, f"--n-rows must be greater than 0."
        end_idx = args.n_rows + start_idx
    else:
        end_idx = len(input_data)
    logging.info(f"Processing {end_idx - start_idx} rows.")
    #######################
    #                     #
    #    prepare model    #
    #                     #
    #######################
    logging.info("Loading GPT-2 model.")
    model = GPT2Model.from_pretrained("gpt2-medium", device_map=0)  # move to GPU:0
    model.eval()
    logging.info("Loading GPT-2 tokenizer.")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    #################
    #               #
    #    bit enc    #
    #               #
    #################
    if args.overwrite:
        logging.warn(f"Overwriting output file.")
    logging.info(f"Encoding plain-text to bitstring. Output file: {args.output}.")
    os.makedirs(osp.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fp:
        writer = csv.DictWriter(fp, fieldnames=input_fieldnames + [args.dst_col], dialect=dialect)
        writer.writeheader()
        for row in tqdm(input_data[start_idx:end_idx], desc="Plain-To-Bits", dynamic_ncols=True):
            bits_base64 = encode(
                row[args.src_col], size_bits=args.size_bits, ef_rounds=args.ef_rounds
            )
            row[args.dst_col] = bits_base64
            writer.writerow(row)
    logging.info("Done.")
