"""Plain-text to bitstring.

Usage:
    python plain2bit.py -o imdb_s1.csv --n-rows 16000 imdb.csv
"""
import argparse
import csv
import logging
import os
import os.path as osp
import sys
from typing import Any

import torch
from bitstring import BitStream
from tqdm import tqdm

os.environ["HF_HOME"] = f"{osp.dirname(__file__)}/../tmp_saves/hg_cache"
sys.path.append(f"{osp.dirname(osp.abspath(__file__))}/../src")

from transformers import GPT2LMHeadModel, GPT2Tokenizer

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
        default="enc_bits",
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
        "-n",
        "--n-rows",
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
    parser.add_argument(
        "--max-token-length",
        type=int,
        default=64,
        help="Max token length. [Default: 64]",
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
        help="Number of size bits. [Default: 8]",
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
    args = parser.parse_args()
    assert osp.exists(args.input), f"{args.input} does not exist."
    assert args.force or not osp.exists(
        args.output
    ), f"{args.output} already exists. Use --force to overwrite."
    assert args.size_bits > 0, f"--size-bits must be greater than 0."
    assert args.ef_rounds >= 0, f"--ef-rounds must be greater than or equal to 0."

    return args


@torch.no_grad()
def encode(
    args: argparse.Namespace,
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    plaintext: str,
) -> str:
    """Encode plaintext to bitstring.

    Args:
        args (argparse.Namespace): Command-line arguments.
        model (GPT2Model): GPT-2 model.
        tokenizer (GPT2Tokenizer): GPT-2 tokenizer.
        plaintext (str): Plain-text.

    Returns:
        str: Bitstring encoded in Base64 format.
    """
    device = model.device
    token_ids: torch.Tensor = tokenizer(
        plaintext,
        return_tensors="pt",
        max_length=args.max_token_length,
        truncation=True,
    ).input_ids.to(
        device
    )  # (1, seq_len)
    bs = codec.encode_token_ids(
        model,
        token_ids,
        add_bos_token=True,
        max_bits_len=(2**args.size_bits - args.size_bits - 1),
    )
    # pad to multiple of 8 bits
    bs = codec.wrap_bits(bs, size_bits=args.size_bits, ef_rounds=args.ef_rounds)
    bs = BitStream(bs)
    bs.append("0b0" * (8 - len(bs) % 8))
    return codec.bits2base64(bs)


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
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

    logging.info("Loading GPT-2 model.")
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium", device_map=0)  # move to GPU:0
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
    logging.info(f"Encoding plaintext to bitstring. Output file: {args.output}.")
    os.makedirs(osp.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fp:
        writer = csv.DictWriter(fp, fieldnames=input_fieldnames + [args.dst_col])
        writer.writeheader()
        for row in tqdm(input_data[start_idx:end_idx], desc="Plain-To-Bits", dynamic_ncols=True):
            bits_base64 = encode(args, model, tokenizer, row[args.src_col])
            row[args.dst_col] = bits_base64
            writer.writerow(row)
    logging.info("Done.")
