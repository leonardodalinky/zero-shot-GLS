"""Train the RNN-Stega model."""
import argparse
import logging
import os
import os.path as osp
from functools import partial

TMP_SAVES_DIR = f"{osp.dirname(__file__)}/../../tmp_saves"
os.environ["HF_HOME"] = f"{TMP_SAVES_DIR}/hg_cache"

import accelerate
import torch
import transformers as tr
from torch.utils.data import DataLoader
from torchinfo import summary as model_summary
from tqdm import tqdm

tqdm = partial(tqdm, dynamic_ncols=True)

# isort: split
from CSVDataset import CSVDataset
from model import RNNStega


def parse_args():
    parser = argparse.ArgumentParser(
        "Train RNN-Stega model to fit the data.",
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
    parser.add_argument("--epoch", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for training.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training.")
    parser.add_argument(
        "--max-token-length",
        type=int,
        default=128,
        help="Maximum token length for training.",
    )
    parser.add_argument("--save-interval", type=int, default=20, help="Save interval.")
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
    #    Validating    #
    #                  #
    ####################
    args = parser.parse_args()
    assert osp.exists(args.input), f"Input file {args.input} does not exist."

    return args


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
    tokenizer = tr.AutoTokenizer.from_pretrained("facebook/opt-125m")
    ####################
    #                  #
    #    load model    #
    #                  #
    ####################
    model: RNNStega = RNNStega(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
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
    for epoch in tqdm(range(1, args.epoch + 1), desc="Epoch"):
        loss_list: list[float] = []
        for batch in tqdm(dataloader, desc="Batch"):
            # tokenize
            text: list[str] = batch["text"]
            token_ids: torch.Tensor = tokenizer(
                text,
                padding="longest",
                truncation=True,
                max_length=args.max_token_length,
                return_tensors="pt",
            ).input_ids
            token_ids = token_ids.to(device)

            loss = model.forward_train(token_ids)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
        logging.info(f"Epoch {epoch}: loss = {sum(loss_list) / len(loss_list)}")
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
