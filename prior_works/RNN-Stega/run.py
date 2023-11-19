"""Run NLS for generating stego-text.

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

import accelerate
import torch
from tqdm import tqdm

# isort: split
from model import RNNStega

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
    ####################
    #                  #
    #    Model Args    #
    #                  #
    ####################
    parser.add_argument("--embed-dim", type=int, default=384, help="Embedding dimension.")
    parser.add_argument("--hidden-dim", type=int, default=768, help="Hidden dimension.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate.")
    ################
    #              #
    #    params    #
    #              #
    ################
    parser.add_argument(
        "--max-bpw",
        type=int,
        default=3,
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
    # different mode check

    return args


@torch.no_grad()
def encrypt(
    model: tr.AutoModelForCausalLM,
    tokenizer: tr.AutoTokenizer,
    context: str,
    bs_base64: str,
    seed: int,
    max_bpw: int,
    sentence_id: int | None = None,
) -> tuple[str, float]:
    device = model.device
    # decode base64
    bs = codec.base642bits(bs_base64)
    message: list[int] = [int(c) for c in bs.bin]
    with prompt_gen.random_state(seed):
        context_ids = encode_context(context, tokenizer)
        output_ids, avg_nll = encode_huffman(
            model=model,
            enc=tokenizer,
            message=message,
            context=context_ids,
            bits_per_word=max_bpw,
            finish_sent=complete_sent,
            device=device,
        )[:2]
        ppl = 2**avg_nll
        output_text = tokenizer.decode(output_ids)
        return output_text, ppl


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
    model: RNNStega = RNNStega(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    accelerator = accelerate.Accelerator()
    model = model.to(accelerator.device)
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
            fieldnames=input_fieldnames + [args.dst_col, args.ppl_col, args.seed_col],
        )
        writer.writeheader()
        for row_idx, row in enumerate(
            tqdm(input_data[start_idx:end_idx], desc="RNNStega-Bits-To-Stego", dynamic_ncols=True)
        ):
            seed = seeds[row_idx]
            stegotext, ppl = encrypt(
                model,
                tokenizer,
                bs_base64=row[args.src_col],
                seed=seed,
                sentence_id=row.get("sentence_id"),
                max_bpw=args.max_bpw,
            )
            # remove all newlines
            stegotext = stegotext.replace("\n", " ")
            stegotext = stegotext.replace("\r", " ")
            row[args.dst_col] = stegotext
            row[args.seed_col] = seed
            row[args.ppl_col] = f"{ppl:.4f}"
            writer.writerow(row)
    logging.info("Done.")
