"""Run RNN-Stega for generating stego-text.

Usage:
    TODO
"""
import argparse
import csv
import logging
import math
import os
import os.path as osp
import random
import sys
from typing import Any

import accelerate
import torch
import torch.nn.functional as F
from tqdm import tqdm

# isort: split
from model import ADG

os.environ["HF_HOME"] = f"{osp.dirname(__file__)}/../../tmp_saves/hg_cache"
sys.path.append(f"{osp.dirname(osp.abspath(__file__))}/../../src")

import transformers as tr

import codec
import prompt_gen
from p_utils import seed_everything


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
    parser.add_argument("--ckpt-dir", type=str, required=True, help="Directory to load the model.")
    parser.add_argument(
        "-s",
        "--src-col",
        type=str,
        default="enc_bits",
        help="Column name of source encoded bitstring.",
    )
    parser.add_argument(
        "-d",
        "--dst-col",
        type=str,
        default="stegotext",
        help="Column name of destination stegotext.",
    )
    parser.add_argument(
        "--seed-col",
        type=str,
        default="enc_seed",
        help="Column name of the seed used to generate stegotext.",
    )
    parser.add_argument(
        "--ppl-col",
        type=str,
        default="ppl",
        help="Column name of the ppl of the stegotext.",
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
        required=True,
        help="Output .csv file.",
    )
    parser.add_argument(
        "-n",
        "--n-rows",
        type=int,
        default=4000,
        help="Number of rows to process. Defaults to all.",
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
    parser.add_argument(
        "--seed-gen-seed",
        type=int,
        default=2023,
        help="Seed to generate seeds.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=160,
        help="Max new tokens to be generated in hiding process.",
    )
    ####################
    #                  #
    #    Model Args    #
    #                  #
    ####################
    parser.add_argument("--embed-dim", type=int, default=384, help="Embedding dimension.")
    parser.add_argument("--hidden-dim", type=int, default=768, help="Hidden dimension.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate.")
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

    return args


def prepare_start_ids(
    model: ADG,
    tokenizer: tr.AutoTokenizer,
) -> torch.Tensor:
    ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=model.device)  # (1, 1)
    logits: torch.Tensor = model(ids)[0, 0]  # (V,)
    sorted_logits_indices = torch.argsort(logits, descending=True)
    # randomly sample one token from the first 1000
    sorted_logits_indices = sorted_logits_indices[:1000]
    rand_idx = torch.randint(0, len(sorted_logits_indices), (1,))
    rand_token_id = sorted_logits_indices[rand_idx]
    start_ids = torch.tensor(
        [
            [
                tokenizer.bos_token_id,  # bos
                rand_token_id.item(),  # random token
            ]
        ],
        dtype=torch.long,
        device=model.device,
    )
    return start_ids


# e.g. [0, 1, 1, 1] looks like 1110=14
def bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += bit * (2**i)
    return res


def int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ("{0:0%db}" % num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]


def near(alist, anum):
    up = len(alist) - 1
    if up == 0:
        return 0
    bottom = 0
    while up - bottom > 1:
        index = int((up + bottom) / 2)
        if alist[index] < anum:
            up = index
        elif alist[index] > anum:
            bottom = index
        else:
            return index
    if up - bottom == 1:
        if alist[bottom] - anum < anum - up:
            index = bottom
        else:
            index = up
    return index


@torch.no_grad()
def encrypt(
    model: ADG,
    tokenizer: tr.AutoTokenizer,
    bs_base64: str,
    seed: int,
    sentence_id: int | None = None,
    max_new_tokens: int = 160,
) -> tuple[str, float, int]:
    device = model.device
    # decode base64
    bs = codec.base642bits(bs_base64)
    bit_stream: str = bs.bin
    with prompt_gen.random_state(seed):
        bit_index = 0
        input_ids = prepare_start_ids(model, tokenizer=tokenizer)  # (1, S)
        nll_list: list[float] = []
        for _ in range(max_new_tokens):
            logits: torch.Tensor = model(input_ids, logits=True)[0, -1]  # (V,)
            logits = logits.double()
            logits[tokenizer.bos_token_id] = -10
            logits[tokenizer.eos_token_id] = -10
            logits[tokenizer.pad_token_id] = -10
            probs = F.softmax(logits, dim=-1)  # (V,)
            original_probs = probs.clone()
            probs, indices = probs.sort(descending=True)
            # start recursion
            bit_tmp = 0
            while probs[0] <= 0.5:
                # embedding bit
                bit = 1
                while (1 / 2 ** (bit + 1)) > probs[0]:
                    bit += 1
                mean = 1 / 2**bit
                # dp
                probs = probs.tolist()
                indices = indices.tolist()
                result = []
                for i in range(2**bit):
                    result.append([[], []])
                for i in range(2**bit - 1):
                    result[i][0].append(probs[0])
                    result[i][1].append(indices[0])
                    del probs[0]
                    del indices[0]
                    while sum(result[i][0]) < mean:
                        delta = mean - sum(result[i][0])
                        index = near(probs, delta)
                        if probs[index] - delta < delta:
                            result[i][0].append(probs[index])
                            result[i][1].append(indices[index])
                            del probs[index]
                            del indices[index]
                        else:
                            break
                    mean = sum(probs) / (2**bit - i - 1)
                result[2**bit - 1][0].extend(probs)
                result[2**bit - 1][1].extend(indices)
                # read secret message
                bit_embed = [
                    int(_) for _ in bit_stream[bit_index + bit_tmp : bit_index + bit_tmp + bit]
                ]
                int_embed = bits2int(bit_embed)
                # updating
                probs = torch.FloatTensor(result[int_embed][0]).to(device)
                indices = torch.LongTensor(result[int_embed][1]).to(device)
                probs = probs / probs.sum()
                probs, _ = probs.sort(descending=True)
                indices = indices[_]
                bit_tmp += bit

            # terminate
            gen = int(indices[int(torch.multinomial(probs, 1))])
            nll_list.append(-math.log2(original_probs[gen].item()))
            input_ids = torch.cat(
                [input_ids, torch.tensor([[gen]], dtype=torch.long, device=device)], dim=1
            )
            bit_index += bit_tmp
            if bit_index >= len(bit_stream):
                break

        avg_nll = sum(nll_list) / len(nll_list)
        ppl = 2**avg_nll
        return tokenizer.decode(input_ids[0, 1:].tolist()), ppl, bit_index


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    logging.info(f"Args: {args}")
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
    assert (
        args.seed_col not in reader.fieldnames
    ), f"Seed column '{args.seed_col}' already in {args.input}."
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

    tokenizer = tr.AutoTokenizer.from_pretrained("facebook/opt-125m")
    model: ADG = ADG(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_token_id=tokenizer.pad_token_id,
    )
    accelerator = accelerate.Accelerator(mixed_precision="fp16")
    model = model.to(accelerator.device)
    logging.info(f"Loading model from {args.ckpt_dir}.")
    model = accelerator.prepare(model)
    accelerator.load_state(args.ckpt_dir)
    model.eval()
    #################
    #               #
    #    bit enc    #
    #               #
    #################
    if args.force and osp.exists(args.output):
        logging.warning(f"Overwriting output file.")
    logging.info(f"Encrypt bitstring to stegotext. Output file: {args.output}.")
    os.makedirs(osp.dirname(osp.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=input_fieldnames
            + [args.dst_col, args.ppl_col, args.seed_col, args.used_bits_col],
        )
        writer.writeheader()
        for row_idx, row in enumerate(
            tqdm(input_data[start_idx:end_idx], desc="ADG-Bits-To-Stego", dynamic_ncols=True)
        ):
            seed = seeds[row_idx]
            stegotext, ppl, used_bits = encrypt(
                model,
                tokenizer,
                bs_base64=row[args.src_col],
                seed=seed,
                sentence_id=row.get("sentence_id"),
                max_new_tokens=args.max_new_tokens,
            )
            # remove all newlines
            stegotext = stegotext.replace("\n", " ")
            stegotext = stegotext.replace("\r", " ")
            row[args.dst_col] = stegotext
            row[args.seed_col] = seed
            row[args.ppl_col] = f"{ppl:.4f}"
            row[args.used_bits_col] = used_bits
            writer.writerow(row)
    logging.info("Done.")
