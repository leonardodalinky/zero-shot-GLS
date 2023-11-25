"""Train the RNN-Stega model."""
import argparse
import logging
import os
import os.path as osp
from functools import partial

TMP_SAVES_DIR = f"{osp.dirname(__file__)}/../../tmp_saves"
os.environ["HF_HOME"] = f"{TMP_SAVES_DIR}/hg_cache"

import accelerate
import numpy as np
import torch
import transformers as tr
from torch.utils.data import DataLoader
from torchinfo import summary as model_summary
from tqdm import tqdm

tqdm = partial(tqdm, dynamic_ncols=True)

# isort: split
from CSVDataset import create_train_test_datasets
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
    parser.add_argument("--train-ratio", type=float, default=0.85, help="Train ratio.")
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
    accelerator = accelerate.Accelerator(mixed_precision="fp16")
    accelerate.utils.set_seed(args.seed)
    device = accelerator.device
    ###################
    #                 #
    #    load data    #
    #                 #
    ###################
    logging.info(f"Loading input data: {args.input}.")
    train_dataset, test_dataset = create_train_test_datasets(
        args.input, args.data_col, train_ratio=args.train_ratio, seed=args.seed
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
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
        pad_token_id=tokenizer.pad_token_id,
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
    model, train_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, test_dataloader, optimizer, lr_scheduler
    )

    ##################
    #                #
    #    Training    #
    #                #
    ##################
    logging.info("Start training. Good luck!")
    best_loss = np.inf
    best_epoch = 0
    for epoch in tqdm(range(1, args.epoch + 1), desc="Epoch"):
        train_loss_list: list[float] = []
        test_loss_list: list[float] = []
        #####################
        #                   #
        #    train phase    #
        #                   #
        #####################
        model.train()
        for batch in tqdm(train_dataloader, desc="Train-Batch"):
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
            train_loss_list.append(loss.item())

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
        ####################
        #                  #
        #    test phase    #
        #                  #
        ####################
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Test-Batch"):
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
                test_loss_list.append(loss.item())
        logging.info(
            f"Epoch {epoch}: train_loss = {np.mean(train_loss_list):.4f}, test_loss = {np.mean(test_loss_list):.4f}"
        )
        # regular save
        if epoch % args.save_interval == 0:
            ckpt_dir = osp.join(args.save_dir, f"epoch_{epoch}")
            logging.info(f"Saving model to {ckpt_dir}.")
            accelerator.save_state(ckpt_dir)
        # best save
        if (avg_test_loss := np.mean(test_loss_list)) < best_loss:
            best_loss = avg_test_loss
            best_epoch = epoch
            best_dir = osp.join(args.save_dir, "best")
            logging.info(f"Saving best model to {args.save_dir} at Epoch {epoch}.")
            accelerator.save_state(best_dir)
        else:
            logging.info(f"Best model is still at Epoch {best_epoch}.")

    # indicate best model epoch
    with open(osp.join(args.save_dir, "best_epoch.txt"), "w") as fp:
        fp.write(f"{best_epoch}\n")

    # create soft link to last checkpoint
    logging.info("Creating soft link to last checkpoint.")
    softlink_dir = osp.join(args.save_dir, "ckpt")
    if osp.exists(softlink_dir):
        os.remove(softlink_dir)
    os.symlink(f"epoch_{args.epoch}", softlink_dir)

    logging.info("Done.")


if __name__ == "__main__":
    main()
