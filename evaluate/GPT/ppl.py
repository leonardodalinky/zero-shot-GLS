"""Compute PPL based on trained GPT-2 model."""
import argparse
import json
import logging
import os
import os.path as osp
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["HF_HOME"] = f"{osp.dirname(__file__)}/../../tmp_saves/hg_cache"
sys.path.append(f"{osp.dirname(osp.abspath(__file__))}/../../src")

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from p_utils import seed_everything


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
        "--data-col",
        type=str,
        required=True,
        help="Name of the column containing the textual data.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output .json file.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to the directory containing the trained model.",
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
    parser.add_argument(
        "--max-token-length",
        type=int,
        default=128,
        help="Maximum token length for training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size.",
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


def perplexity(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    texts: list[str],
    max_token_length: int,
) -> list[float]:
    """Compute perplexity of a given text."""
    device = model.device
    batch = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_token_length,
    ).to(device)
    input_ids: torch.Tensor = batch.input_ids  # (B, L)
    attn_mask: torch.Tensor = batch.attention_mask  # (B, L)
    logits: torch.Tensor = model(**batch).logits  # (B, L, V)
    probs = torch.softmax(logits, dim=-1)  # (B, L, V)
    probs = probs.clamp_(min=1e-10)  # (B, L, V)
    _input_ids = input_ids[:, 1:]  # (B, L-1)
    _probs = probs[:, :-1]  # (B, L-1, V)
    _attn_mask = attn_mask[:, 1:]  # (B, L-1)
    assert _input_ids.shape[:2] == _probs.shape[:2]

    # compute ppl
    tmp = torch.gather(_probs, dim=-1, index=_input_ids.unsqueeze(-1))  # (B, L-1, 1)
    tmp = tmp.squeeze(-1)  # (B, L-1)
    log2_tmp = torch.log2(tmp)  # (B, L-1)
    log2_tmp *= _attn_mask  # (B, L-1)

    ret = 2 ** (-log2_tmp.sum(dim=-1) / _attn_mask.sum(dim=-1))  # (B,)
    return ret.tolist()


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
    df = pd.read_csv(args.input)
    assert args.data_col in df.columns, f"Data Column '{args.data_col}' not in {args.input}."
    input_data: list[str] = df[args.data_col].to_list()
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
    dataloader = DataLoader(
        input_data[start_idx:end_idx],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    #######################
    #                     #
    #    prepare model    #
    #                     #
    #######################
    logging.info(f"Loading GPT-2 model from {args.model_dir}.")
    model = GPT2LMHeadModel.from_pretrained(
        args.model_dir, device_map=0, local_files_only=True
    )  # move to GPU:0
    model.eval()
    logging.info(f"Loading GPT-2 tokenizer from {args.model_dir}.")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_dir)
    #################
    #               #
    #    metrics    #
    #               #
    #################
    if args.force and osp.exists(args.output):
        logging.warning(f"Overwriting output file.")
    logging.info(f"Compute metrics. Output file: {args.output}.")
    ppl_list: list[float] = []
    with torch.no_grad():
        for texts in tqdm(dataloader, desc="PPL", dynamic_ncols=True):
            # texts: list[str]
            ppls = perplexity(model, tokenizer, texts, args.max_token_length)
            ppl_list.extend(ppls)

    ppl_mean = np.mean(ppl_list)
    print("ppl_mean:", ppl_mean)
    os.makedirs(osp.dirname(osp.abspath(args.output)), exist_ok=True)
    logging.info(f"Saving metrics to {args.output}.")
    with open(args.output, "w") as fp:
        json.dump(
            {
                "ppl_mean": ppl_mean,
            },
            fp,
        )
    logging.info("Done.")
