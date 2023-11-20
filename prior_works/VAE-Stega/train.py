"""Train the RNN-Stega model."""
import argparse
import logging
import math
import os
import os.path as osp
from functools import partial

import numpy as np

TMP_SAVES_DIR = f"{osp.dirname(__file__)}/../../tmp_saves"
os.environ["HF_HOME"] = f"{TMP_SAVES_DIR}/hg_cache"

import accelerate
import torch
import torch.nn.functional as F
import transformers as tr
from torch.utils.data import DataLoader
from torchinfo import summary as model_summary
from tqdm import tqdm

tqdm = partial(tqdm, dynamic_ncols=True)

# isort: split
from CSVDataset import CSVDataset
from model import VAE

WORD_DROPOUT = 0.5


def parse_args():
    parser = argparse.ArgumentParser(
        "Train VAE-Stega model to fit the data.",
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
        default="plaintext",
        help="Name of the column containing the data.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory to save the trained model.",
    )
    ######################
    #                    #
    #    Dataset Args    #
    #                    #
    ######################
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    #######################
    #                     #
    #    Training Args    #
    #                     #
    #######################
    parser.add_argument("--epoch", type=int, default=20, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for training.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument(
        "--max-token-length",
        type=int,
        default=128,
        help="Maximum token length for training.",
    )
    parser.add_argument("--save-interval", type=int, default=5, help="Save interval.")
    ####################
    #                  #
    #    Model Args    #
    #                  #
    ####################
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--num-z", type=int, default=128, help="Dimension of z latent.")
    ####################
    #                  #
    #    Validating    #
    #                  #
    ####################
    args = parser.parse_args()
    assert osp.exists(args.input), f"Input file {args.input} does not exist."

    return args


def create_generator_input(x: torch.Tensor, tokenizer: tr.AutoTokenizer):
    G_inp = x[
        :, : x.size(1) - 1
    ].clone()  # input for generator should exclude last word of sequence

    r = torch.rand_like((G_inp))
    # Perform word_dropout according to random values (r) generated for each word
    for i in range(G_inp.size(0)):
        for j in range(1, G_inp.size(1)):
            if r[i, j] < WORD_DROPOUT and G_inp[i, j] not in [
                tokenizer.pad_token_id,
                tokenizer.cls_token_id,
            ]:
                G_inp[i, j] = tokenizer.mask_token_id

    return G_inp


def train_batch(
    vae: VAE,
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    G_inp: torch.Tensor,
    step: int,
):
    logit, _, kld = vae(input_ids, attn_mask, G_inp, None, None)
    # logit: (B, S, V - 1)
    # NOTE: the `G_inp` has already cut the last token
    logit = logit.reshape(-1, logit.size(2))
    input_ids = input_ids[:, 1:]  # target for generator should exclude first word of sequence
    input_ids = input_ids.reshape(-1)
    rec_loss = F.cross_entropy(logit, input_ids)
    kld_coef = (math.tanh((step - 15000) / 1000) + 1) / 2
    loss = 7 * rec_loss + kld_coef * kld
    return loss, rec_loss, kld


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    accelerator = accelerate.Accelerator()
    accelerate.utils.set_seed(args.seed)
    device = accelerator.device
    ###################
    #                 #
    #    load data    #
    #                 #
    ###################
    logging.info(f"Loading input data: {args.input}.")
    dataset = CSVDataset(args.input, args.data_col)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    ################################
    #                              #
    #    load vocab & tokenizer    #
    #                              #
    ################################
    tokenizer = tr.AutoTokenizer.from_pretrained("distilbert-base-uncased")
    ####################
    #                  #
    #    load model    #
    #                  #
    ####################
    model: VAE = VAE(n_layer=args.num_layers, n_z=args.num_z)
    model.to(device)
    logging.info(f"Model summary:\n{model_summary(model)}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epoch,
        eta_min=1e-5,
    )

    #################
    #               #
    #    prapare    #
    #               #
    #################
    model, dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, dataloader, optimizer, lr_scheduler
    )

    ##################
    #                #
    #    Training    #
    #                #
    ##################
    logging.info("Start training. Good luck!")
    model.train()
    step = 0
    for epoch in tqdm(range(1, args.epoch + 1), desc="Epoch"):
        loss_list: list[float] = []
        rec_loss_list: list[float] = []
        kl_loss_list: list[float] = []
        for batch in tqdm(dataloader, desc="Batch"):
            # tokenize
            text: list[str] = batch["text"]
            tokenized = tokenizer(
                text,
                padding="longest",
                truncation=True,
                max_length=args.max_token_length - 1,
                add_special_tokens=False,
                return_tensors="pt",
            )
            tokenized = tokenized.to(device)
            input_ids = tokenized.input_ids
            # append [CLS]
            input_ids = torch.cat(
                [
                    torch.full_like(input_ids[:, :1], tokenizer.cls_token_id),
                    input_ids,
                ],
                dim=1,
            )

            G_inp = create_generator_input(input_ids, tokenizer)

            loss, rec_loss, kl_loss = train_batch(
                vae=model,
                input_ids=input_ids,
                attn_mask=tokenized.attention_mask,
                G_inp=G_inp,
                step=step,
            )

            loss_list.append(loss.item())
            rec_loss_list.append(rec_loss.item())
            kl_loss_list.append(kl_loss.item())

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
        logging.info(
            f"Epoch {epoch}: loss = {np.mean(loss_list)}, rec_loss = {np.mean(rec_loss_list)}, kl_loss = {np.mean(kl_loss_list)}"
        )
        if epoch % args.save_interval == 0:
            ckpt_dir = osp.join(args.save_dir, f"epoch_{epoch}")
            logging.info(f"Saving model to {ckpt_dir}.")
            accelerator.save_state(ckpt_dir)
    # create soft link to last checkpoint
    logging.info("Creating soft link to last checkpoint.")
    last_checkpoint_dir = osp.join(args.save_dir, f"epoch_{args.epoch}")
    last_checkpoint_dir = osp.realpath(last_checkpoint_dir)
    softlink_dir = osp.join(args.save_dir, "ckpt")
    if osp.exists(softlink_dir):
        os.remove(softlink_dir)
    os.symlink(last_checkpoint_dir, softlink_dir)

    logging.info("Done.")


if __name__ == "__main__":
    main()
