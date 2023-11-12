"""Bitstring to plain-text.

Usage:
    python bit2plain.py -o imdb_s4.csv --n-rows 16000 imdb_s3.csv
"""
import argparse
import csv
import logging
import os
import os.path as osp
import sys
from typing import Any

import torch
from tqdm import tqdm

os.environ["HF_HOME"] = f"{osp.dirname(__file__)}/../tmp_saves/hg_cache"
sys.path.append(f"{osp.dirname(osp.abspath(__file__))}/../src")

from transformers import GPT2LMHeadModel, GPT2Tokenizer

import codec
from p_utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plain-text to bitstring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=str, help="Input .csv file.")
    parser.add_argument(
        "-s",
        "--src-col",
        type=str,
        default="dec_bits",
        help="Column name of source bitstring.",
    )
    parser.add_argument(
        "-d",
        "--dst-col",
        type=str,
        default="dec_plaintext",
        help="Column name of destination decoded plaintext.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output .csv file.",
    )
    parser.add_argument(
        "-n",
        "--n-rows",
        type=int,
        help="Number of rows to process. Defaults to all rows.",
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
        default=8,
        help="Number of size bits.",
    )
    parser.add_argument(
        "--ef-rounds",
        type=int,
        default=4,
        help="Number of EF rounds.",
    )
    ####################
    #                  #
    #    validating    #
    #                  #
    ####################
    args = parser.parse_args()
    assert osp.exists(args.input), f"{args.input} does not exist."
    assert args.force or not osp.exists(
        args.output
    ), f"{args.output} already exists. Use --force to overwrite."
    assert args.size_bits > 0, f"--size-bits must be greater than 0."
    assert args.ef_rounds >= 0, f"--ef-rounds must be greater than or equal to 0."

    return args


@torch.no_grad()
def decode(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    bs_base64: str,
    size_bits: int,
    ef_rounds: int,
    sentence_id: int | None = None,
) -> str | None:
    """decode plaintext from bitstring.

    Args:
        args (argparse.Namespace): Command-line arguments.
        model (GPT2Model): GPT-2 model.
        tokenizer (GPT2Tokenizer): GPT-2 tokenizer.
        bs_base64 (str): Bitstring encoded in Base64 format.
        sentence_id (int, optional): Sentence ID. Defaults to None. Only used for logging.

    Returns:
        str | None: Plaintext decoded from bitstring. If decode fails, return None.
    """
    # decode base64
    bs = codec.base642bits(bs_base64)
    bs = codec.unwrap_bits(bs, size_bits=size_bits, ef_rounds=ef_rounds)
    try:
        token_ids = codec.decode_bitstream(model, bs, remove_bos_token=True)  # [1, seq_len]
        # decode token_ids
        return tokenizer.decode(token_ids[0, :].tolist())
    except codec.DecodeException as e:
        logging.warning(f"Cannot decode bitstring #{sentence_id}.")
        cur_str = tokenizer.decode(e.token_ids[0, :].tolist())
        return "<ERROR>: " + cur_str


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    seed_everything(42, deterministic=True, warn_only=True)
    ###################
    #                 #
    #    load data    #
    #                 #
    ###################
    logging.info(f"Loading input data: {args.input}.")
    with open(args.input, "r") as fp:
        reader = csv.DictReader(fp)
        input_fieldnames = list(reader.fieldnames)
        input_data: list[dict[str, Any]] = list(reader)
    assert args.src_col in reader.fieldnames, f"Src Column '{args.src_col}' not in {args.input}."
    assert (
        args.dst_col not in reader.fieldnames
    ), f"Dst column '{args.dst_col}' already in {args.input}."
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
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2-medium", device_map=0, local_files_only=True
    )  # move to GPU:0
    model.eval()
    logging.info("Loading GPT-2 tokenizer.")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    #################
    #               #
    #    bit enc    #
    #               #
    #################
    if args.force and osp.exists(args.output):
        logging.warning(f"Overwriting output file.")
    logging.info(f"Decode plaintext from bitstring. Output file: {args.output}.")
    os.makedirs(osp.dirname(osp.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fp:
        writer = csv.DictWriter(fp, fieldnames=input_fieldnames + [args.dst_col])
        writer.writeheader()
        for row in tqdm(input_data[start_idx:end_idx], desc="Bits-To-Plain", dynamic_ncols=True):
            dec_plaintext = decode(
                model,
                tokenizer,
                row[args.src_col],
                size_bits=args.size_bits,
                ef_rounds=args.ef_rounds,
                sentence_id=row.get("sentence_id"),
            )
            row[args.dst_col] = dec_plaintext
            writer.writerow(row)
    logging.info("Done.")
