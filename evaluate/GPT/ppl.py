"""Compute metrics based on trained GPT-2 model."""
import argparse
import csv
import json
import logging
import os
import os.path as osp
import sys
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

os.environ["HF_HOME"] = f"{osp.dirname(__file__)}/../../tmp_saves/hg_cache"
sys.path.append(f"{osp.dirname(osp.abspath(__file__))}/../../src")

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from p_utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(
        "Compute metrics based on trained GPT-2 model.",
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
    assert osp.exists(args.model_dir), f"Model directory {args.model_dir} does not exist."

    return args


def perplexity(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, text: str) -> float | None:
    """Compute perplexity of a given text."""
    device = model.device
    inputs: torch.Tensor = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
    ).input_ids
    assert inputs.dim() == 2
    assert inputs.size(0) == 1
    inputs_gpu = inputs.to(device)  # (1, seq_len)
    logits_gpu: torch.Tensor = model(inputs_gpu).logits
    probs_gpu = torch.softmax(logits_gpu, dim=-1)  # (1, seq_len, vocab_size)
    probs = probs_gpu.cpu()
    sum = 0.0
    if inputs.size(1) <= 1:
        return None

    for input_id, prob in zip(inputs[0, 1:], probs[0, :-1]):
        # input_id: int
        # prob: torch.Tensor of shape (vocab_size,)
        sum += torch.log2(prob[input_id]).item()
    sum_len = inputs.size(1) - 1
    return 2 ** (-sum / sum_len)


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
    assert args.data_col in reader.fieldnames, f"Data Column '{args.data_col}' not in {args.input}."
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
        for row in tqdm(input_data[start_idx:end_idx], desc="Metrics", dynamic_ncols=True):
            ppl = perplexity(model, tokenizer, row[args.data_col])
            if ppl is not None:
                ppl_list.append(ppl)

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
