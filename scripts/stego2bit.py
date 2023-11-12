"""Stego-text to bitstring.

Usage:
    python stego2bit.py -o $DATA_DIR/imdb_s3.csv \
        --force \
        --mode cover \
        --cover $DATA_DIR/imdb.csv \
        --n-cover $n_cover \
        --corpus "IMDB about movies" \
        --threshold $threshold \
        --max-bpw $max_bpw \
        --n-rows 5 \
        $DATA_DIR/imdb_s2.csv
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

from transformers import LlamaForCausalLM, LlamaTokenizer

import codec
import hide_extract
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
        default="stegotext",
        help="Column name of source stegotext.",
    )
    parser.add_argument(
        "-d",
        "--dst-col",
        type=str,
        default="dec_bits",
        help="Column name of destination bistring.",
    )
    parser.add_argument(
        "--cover-col",
        type=str,
        default="plaintext",
        help="Column name of the cover text in `--cover`.",
    )
    parser.add_argument(
        "--seed-col",
        type=str,
        default="enc_seed",
        help="Column name of the seed used to generate stegotext.",
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
    parser.add_argument("--corpus", type=str, default="Unknown", help="Hint for corpus name.")
    ###########################
    #                         #
    #    Generation Config    #
    #                         #
    ###########################
    parser.add_argument("--mode", type=str, required=True, choices=["cover"])
    parser.add_argument("--n-cover", type=int, default=3, help="Number of cover text.")
    ####################
    #                  #
    #    egs params    #
    #                  #
    ####################
    parser.add_argument(
        "--threshold",
        type=float,
        default=2e-3,
        help="Threshold of EGS.",
    )
    parser.add_argument(
        "--max-bpw",
        type=int,
        default=5,
        help="Max bits for each token (word) to hide.",
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
    if args.mode == "cover":
        assert args.cover is not None, "Cover text is required for 'cover' mode."
        assert osp.exists(args.cover), f"{args.cover} does not exist for 'cover' mode."

    return args


@torch.no_grad()
def decrypt(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    prompt: str,
    stegotext: str,
    seed: int,
    egs_threshold: float,
    max_bpw: int,
    sentence_id: int | None = None,
) -> str:
    device = model.device
    with prompt_gen.random_state(seed):
        prompt_ids: torch.Tensor = tokenizer(prompt, return_tensors="pt").input_ids.to(
            device
        )  # (1, seq_len)
        hide_ids: torch.Tensor = tokenizer(prompt + stegotext, return_tensors="pt").input_ids.to(
            device
        )  # (1, seq_len)
        bs, is_succeed = hide_extract.extract_bits_with_prompt_ids_by_egs(
            model,
            prompt_ids=prompt_ids,
            hide_ids=hide_ids,
            threshold=egs_threshold,
            max_bpw=max_bpw,
        )
        if not is_succeed:
            logging.warning(f"Cannot decode bitstring #{sentence_id}.")

        bs.append("0b0" * (8 - len(bs) % 8))

        return codec.bits2base64(bs)


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
    assert args.seed_col in reader.fieldnames, f"Seed column '{args.seed_col}' not in {args.input}."
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

    model_name = "TheBloke/Llama-2-7B-chat-GPTQ"
    logging.info(f"Loading {model_name} tokenizer.")
    tokenizer = LlamaTokenizer.from_pretrained(model_name, legacy=True)
    logging.info(f"Loading {model_name} model.")
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        local_files_only=True,
        trust_remote_code=False,
        revision="main",
        device_map=0,
    )
    model.eval()
    #################
    #               #
    #    bit dec    #
    #               #
    #################
    if args.force and osp.exists(args.output):
        logging.warning(f"Overwriting output file.")
    logging.info(f"Decrypt bitstring from stegotext. Output file: {args.output}.")
    os.makedirs(osp.dirname(osp.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fp, prompt_gen.gen_prompt_ctx(
        mode=args.mode, cover=args.cover, cover_col=args.cover_col
    ) as gen_prompt:
        writer = csv.DictWriter(fp, fieldnames=input_fieldnames + [args.dst_col])
        writer.writeheader()
        for row in tqdm(input_data[start_idx:end_idx], desc="Stego-To-Bits", dynamic_ncols=True):
            seed = int(row[args.seed_col])
            prompt = gen_prompt(
                n_ctx=args.n_cover,
                seed=seed,
                corpus=args.corpus,
            )
            dec_bits = decrypt(
                model,
                tokenizer,
                prompt=prompt,
                stegotext=row[args.src_col],
                seed=seed,
                sentence_id=row.get("sentence_id"),
                egs_threshold=args.threshold,
                max_bpw=args.max_bpw,
            )
            row[args.dst_col] = dec_bits
            writer.writerow(row)
    logging.info("Done.")
