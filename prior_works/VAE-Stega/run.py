"""Run VAE-Stega for generating stego-text.

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
import Huffman_Encoding
from model import VAE

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
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--num-z", type=int, default=128, help="Dimension of z latent.")
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


def prepare_start(
    model: VAE, tokenizer: tr.AutoTokenizer, n_layers: int, n_z: int, hidden_size: int
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    z = torch.randn([1, n_z], device=model.device)  # (1, Z)
    h_0 = torch.zeros((n_layers, 1, hidden_size), device=model.device)
    c_0 = torch.zeros((n_layers, 1, hidden_size), device=model.device)
    G_hidden = (h_0, c_0)
    G_inp = torch.tensor(
        [[tokenizer.cls_token_id]], dtype=torch.long, device=model.device
    )  # (1, 1)

    # logits: (1, 1, V)
    logits, G_hidden = model.generator(G_inp, z, G_hidden)
    sorted_logits_indices = torch.argsort(logits, descending=True)
    # randomly sample one token from the first 1000
    sorted_logits_indices = sorted_logits_indices[:1000]
    rand_idx = torch.randint(0, len(sorted_logits_indices), (1,))
    rand_token_id = sorted_logits_indices[rand_idx]
    start_ids = torch.tensor(
        [
            [
                tokenizer.cls_token_id,  # bos
                rand_token_id.item(),  # random token
            ]
        ],
        dtype=torch.long,
        device=model.device,
    )
    return start_ids, z, G_hidden


@torch.no_grad()
def encrypt(
    model: VAE,
    tokenizer: tr.AutoTokenizer,
    bs_base64: str,
    seed: int,
    max_bpw: int,
    sentence_id: int | None = None,
    max_new_tokens: int = 160,
) -> tuple[str, float, int]:
    device = model.device
    # decode base64
    bs = codec.base642bits(bs_base64)
    message: str = bs.bin
    with prompt_gen.random_state(seed):
        bit_index = 0
        input_ids, z, G_hidden = prepare_start(
            model,
            tokenizer=tokenizer,
            n_layers=model.generator.n_layer,
            n_z=model.n_z,
            hidden_size=model.generator.hidden_size,
        )
        nll_list: list[float] = []
        for _ in range(max_new_tokens):
            # TODO: VAE infer
            logits, G_hidden = model.generator(input_ids, z, G_hidden)
            logits = logits.double()
            logits[tokenizer.cls_token_id] = -10
            logits[tokenizer.sep_token_id] = -10
            logits[tokenizer.pad_token_id] = -10
            probs = F.softmax(logits, dim=-1)  # (V,)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            sorted_probs: list[float] = sorted_probs.tolist()[: 2**max_bpw]
            sorted_indices: list[int] = sorted_indices.tolist()[: 2**max_bpw]
            # huffman
            nodes = Huffman_Encoding.createNodes(sorted_probs)
            root = Huffman_Encoding.createHuffmanTree(nodes)
            codes = Huffman_Encoding.huffmanEncoding(nodes, root)

            for i in range(2**max_bpw):
                if message[bit_index : bit_index + 1 + i] in codes:
                    code_index = codes.index(message[bit_index : bit_index + 1 + i])
                    nll_list.append(-math.log2(sorted_probs[code_index]))
                    gen_word_id = sorted_indices[code_index]
                    bit_index = bit_index + 1 + i
                    input_ids = torch.cat(
                        [
                            input_ids,
                            torch.tensor([[gen_word_id]], dtype=torch.long, device=device),
                        ],
                        dim=-1,
                    )
                    break
            else:
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
    model: VAE = VAE(
        n_layer=args.num_layers,
        n_z=args.num_z,
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
            tqdm(input_data[start_idx:end_idx], desc="VAE-Bits-To-Stego", dynamic_ncols=True)
        ):
            seed = seeds[row_idx]
            stegotext, ppl, used_bits = encrypt(
                model,
                tokenizer,
                bs_base64=row[args.src_col],
                seed=seed,
                sentence_id=row.get("sentence_id"),
                max_bpw=args.max_bpw,
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
