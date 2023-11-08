"""Bitstring to stego-text.

Usage:
    TODO
"""
import argparse
import csv
import logging
import os
import os.path as osp
import random
import sys
from typing import Any

import torch
from bitstring import BitStream
from tqdm import tqdm

os.environ["HF_HOME"] = f"{osp.dirname(__file__)}/../tmp_saves/hg_cache"
sys.path.append(f"{osp.dirname(osp.abspath(__file__))}/../src")

from transformers import LlamaForCausalLM, LlamaTokenizer

import codec
import hide_extract
import prompt_gen
from p_utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Bitstring to stegotext based on cover text.")
    ###############
    #             #
    #    I / O    #
    #             #
    ###############
    parser.add_argument("input", type=str, help="Input .csv file.")
    parser.add_argument(
        "-c",
        "--cover",
        type=str,
        default=None,
        help="Cover text .csv file. Only used in 'cover' mode.",
    )
    parser.add_argument(
        "-s",
        "--src-col",
        type=str,
        default="enc_bits",
        help="Column name of source encoded bitstring. [Default: enc_bits]",
    )
    parser.add_argument(
        "-d",
        "--dst-col",
        type=str,
        default="stegotext",
        help="Column name of destination stegotext. [Default: stegotext]",
    )
    parser.add_argument(
        "--cover-col",
        type=str,
        default="plaintext",
        help="Column name of the cover text in `--cover`. [Default: plaintext]",
    )
    parser.add_argument(
        "--seed-col",
        type=str,
        default="enc_seed",
        help="Column name of the seed used to generate stegotext. [Default: enc_seed]",
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
    ###########################
    #                         #
    #    Generation Config    #
    #                         #
    ###########################
    parser.add_argument("--mode", type=str, required=True, choices=["cover"])
    parser.add_argument("--n-cover", type=int, default=10, help="Number of cover text")
    parser.add_argument(
        "--max-token-length",
        type=int,
        default=128,
        help="Max token length of generated stegotext. [Default: 128]",
    )
    parser.add_argument(
        "--seed-gen-seed",
        type=int,
        default=2023,
        help="Seed to generate seeds.",
    )
    # TODO: egs params
    parser.add_argument("--TODO")
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
    # different mode check
    if args.mode == "cover":
        assert osp.exists(args.cover), f"{args.cover} does not exist for 'cover' mode."

    return args


@torch.no_grad()
def encrypt(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    prompt: str,
    bs_base64: str,
    seed: int,
) -> str:
    device = model.device
    # decode base64
    bs = codec.base642bits(bs_base64)
    with prompt_gen.random_state(seed):
        prompt_ids: torch.Tensor = tokenizer(prompt, return_tensors="pt").input_ids.to(
            device
        )  # (1, seq_len)
        # TODO
        ...
    ...


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
    #    Generate Seed    #
    #                     #
    #######################
    seed_gen = random.Random(args.seed_gen_seed)
    seeds = [seed_gen.randint(0, 2**32 - 1) for _ in range(end_idx - start_idx)]
    #######################
    #                     #
    #    prepare model    #
    #                     #
    #######################
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

    logging.info("Loading LLaMA2 tokenizer.")
    tokenizer = LlamaTokenizer.from_pretrained("TheBloke/Llama-2-13B-chat-GPTQ")
    logging.info("Loading LLaMA2 model.")
    model = LlamaForCausalLM.from_pretrained(
        "TheBloke/Llama-2-13B-chat-GPTQ",
        local_files_only=True,
        trust_remote_code=False,
        revision="gptq-4bit-32g-actorder_True",
        device_map=0,
    )
    model.eval()
    #################
    #               #
    #    bit enc    #
    #               #
    #################
    if args.force and osp.exists(args.output):
        logging.warning(f"Overwriting output file.")
    logging.info(f"Encoding plaintext to bitstring. Output file: {args.output}.")
    os.makedirs(osp.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fp, prompt_gen.gen_prompt_ctx(
        mode=args.mode, cover=args.cover, cover_col=args.cover_col
    ) as gen_prompt:
        writer = csv.DictWriter(fp, fieldnames=input_fieldnames + [args.dst_col, args.seed_col])
        writer.writeheader()
        for row_idx, row in enumerate(
            tqdm(input_data[start_idx:end_idx], desc="Bits-To-Stego", dynamic_ncols=True)
        ):
            seed = seeds[row_idx]
            prompt = gen_prompt(n=args.n_cover, seed=row[args.seed_col])
            stegotext = encrypt(
                model,
                tokenizer,
                prompt=prompt,
                bs_base64=row[args.src_col],
                seed=seed,
            )
            row[args.dst_col] = stegotext
            row[args.seed_col] = seed
            writer.writerow(row)
    logging.info("Done.")
