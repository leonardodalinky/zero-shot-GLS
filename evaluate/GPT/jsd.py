"""Compute JSD based on trained GPT-2 model."""
import argparse
import json
import logging
import os
import os.path as osp
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["HF_HOME"] = f"{osp.dirname(__file__)}/../../tmp_saves/hg_cache"
sys.path.append(f"{osp.dirname(osp.abspath(__file__))}/../../src")

from train_gpt import gen_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from p_utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(
        "Compute JSD based on trained GPT-2 model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ###############
    #             #
    #    I / O    #
    #             #
    ###############
    parser.add_argument("input", type=str, help="Path to the input .csv dataset file.")
    parser.add_argument(
        "--data-col",
        type=str,
        default="plaintext",
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
        "--model-dir1",
        type=str,
        required=True,
        help="Path to the directory containing the trained model 1.",
    )
    parser.add_argument(
        "--model-dir2",
        type=str,
        required=True,
        help="Path to the directory containing the trained model 2.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite output file.",
    )
    ######################
    #                    #
    #    Dataset Args    #
    #                    #
    ######################
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train-ratio", type=float, default=0.85, help="Train ratio.")
    parser.add_argument("--n-rows", type=int, default=10_000, help="Number of rows to load.")
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


# def gen_dataset(input_path: str, data_col: str) -> Dataset:
#     """Generate a dataset from the input file."""
#     dataset = load_dataset("csv", data_files=input_path, split="train")
#     dataset = dataset.select_columns(data_col)
#     dataset = dataset.rename_column(data_col, "text")
#     return dataset


def kl(input: torch.Tensor, log_target: torch.Tensor) -> torch.Tensor:
    # input: (B, L, V)
    # log_target: (B, L, V)
    # return: (B, L)
    log_input = torch.log2(input)  # (B, L, V)
    return (input * (log_input - log_target)).sum(dim=-1)  # (B, L)


def jensen(
    model1: GPT2LMHeadModel,
    model2: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    texts: list[str],
    max_token_length: int,
) -> list[float] | None:
    """Compute perplexity of a given text."""
    device = model1.device
    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_token_length,
    ).to(device)

    attn_mask: torch.Tensor = batch.attention_mask  # (batch_size, seq_len)
    probs1: torch.Tensor = torch.softmax(model1(**batch).logits, dim=-1)  # (B, L, V)
    probs1 = probs1.clamp_(min=1e-10)
    probs2: torch.Tensor = torch.softmax(model2(**batch).logits, dim=-1)  # (B, L, V)
    probs2 = probs2.clamp_(min=1e-10)
    mix_probs: torch.Tensor = (probs1 + probs2) / 2  # (B, L, V)
    log_mix_probs: torch.Tensor = torch.log2(mix_probs)  # (B, L, V)

    jsd = 0.5 * kl(probs1, log_mix_probs) + 0.5 * kl(probs2, log_mix_probs)  # (B, L)
    jsd *= attn_mask  # (B, L)

    ret = jsd.sum(dim=-1) / attn_mask.sum(dim=-1)  # (B,)
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
    datasets = gen_dataset(args.input, args.data_col, seed=args.seed, train_ratio=args.train_ratio)
    test_dataloader = DataLoader(
        datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    #######################
    #                     #
    #    prepare model    #
    #                     #
    #######################
    logging.info(f"Loading GPT-2 tokenizer from {args.model_dir1}.")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_dir1)
    logging.info(f"Loading GPT-2 model from {args.model_dir1}.")
    model1 = GPT2LMHeadModel.from_pretrained(
        args.model_dir1, device_map=0, local_files_only=True
    )  # move to GPU:0
    model1.eval()
    logging.info(f"Loading GPT-2 model from {args.model_dir2}.")
    model2 = GPT2LMHeadModel.from_pretrained(
        args.model_dir2, device_map=0, local_files_only=True
    )  # move to GPU:0
    model2.eval()
    #################
    #               #
    #    metrics    #
    #               #
    #################
    if args.force and osp.exists(args.output):
        logging.warning(f"Overwriting output file.")
    logging.info(f"Compute metrics. Output file: {args.output}.")
    jsd_list: list[float] = []
    with torch.no_grad():
        for idx, row in tqdm(
            enumerate(test_dataloader),
            desc="JSD",
            total=min(len(test_dataloader), args.n_rows // args.batch_size),
            dynamic_ncols=True,
        ):
            jsds = jensen(model1, model2, tokenizer, row["text"], args.max_token_length)
            jsd_list.extend(jsds)
            if (idx + 1) * args.batch_size >= args.n_rows:
                # stop when we have enough rows
                break

    jsd_mean = np.mean(jsd_list)
    print("jsd_mean:", jsd_mean)
    os.makedirs(osp.dirname(osp.abspath(args.output)), exist_ok=True)
    logging.info(f"Saving metrics to {args.output}.")
    with open(args.output, "w") as fp:
        json.dump(
            {
                "jsd_mean": jsd_mean,
            },
            fp,
        )
    logging.info("Done.")
