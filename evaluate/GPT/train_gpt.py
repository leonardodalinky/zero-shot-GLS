"""Train GPT model to fit the data."""
import argparse
import os
import os.path as osp
from functools import partial

from functional import seq

TMP_SAVES_DIR = f"{osp.dirname(__file__)}/../../tmp_saves"

os.environ["HF_HOME"] = f"{TMP_SAVES_DIR}/hg_cache"

from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    SchedulerType,
    Trainer,
    TrainingArguments,
)

from datasets import DatasetDict, load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        "Train GPT model to fit the data.",
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
    parser.add_argument("--epoch", type=int, default=5, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate for training.")
    parser.add_argument(
        "--max-token-length",
        type=int,
        default=128,
        help="Maximum token length for training.",
    )

    ####################
    #                  #
    #    Validating    #
    #                  #
    ####################
    args = parser.parse_args()
    assert osp.exists(args.input), f"Input file {args.input} does not exist."

    return args


def gen_dataset(
    input_path: str, data_col: str, seed: int, train_ratio: float = 0.85
) -> DatasetDict:
    """Generate a dataset from the input file."""
    assert 0 <= train_ratio <= 1, f"train_ratio must be in [0, 1], got {train_ratio}."
    dataset = load_dataset("csv", data_files=input_path, split="train")
    dataset = dataset.select_columns(data_col)
    dataset = dataset.rename_column(data_col, "text")
    # Train-test split
    datasets = dataset.train_test_split(test_size=1 - train_ratio, shuffle=True, seed=seed)
    return datasets


def tokenize(examples, tokenizer: GPT2Tokenizer, max_token_len: int):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_token_len,
    )


if __name__ == "__main__":
    args = parse_args()
    dataset_name: str = osp.splitext(osp.basename(args.input))[0]
    # load dataset
    datasets: DatasetDict = gen_dataset(
        args.input, args.data_col, seed=args.seed, train_ratio=args.train_ratio
    )
    # load model
    model_name = "gpt2-medium"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    # tokenize data
    tokenized_datasets = datasets.map(
        partial(tokenize, tokenizer=tokenizer, max_token_len=args.max_token_length),
        batched=True,
        remove_columns=["text"],
    )

    train_args = TrainingArguments(
        output_dir=args.save_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        num_train_epochs=args.epoch,
        weight_decay=1e-3,
        warmup_ratio=0.05,
        lr_scheduler_type=SchedulerType.CONSTANT_WITH_WARMUP,
        learning_rate=args.lr,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        seed=args.seed,
        load_best_model_at_end=True,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    ##################
    #                #
    #    training    #
    #                #
    ##################
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    trainer.train()

    # create soft link to last save
    cur_ckpt_dir_list: list[int] = (
        seq(os.listdir(args.save_dir))
        .map(lambda x: osp.join(args.save_dir, x))
        .filter(osp.isdir)
        .map(osp.basename)
        .filter(lambda x: x.startswith("checkpoint-"))
        .map(lambda x: int(x.removeprefix("checkpoint-")))
        .to_list()
    )
    ckpt_name = f"checkpoint-{min(cur_ckpt_dir_list)}"
    new_link_path = osp.join(args.save_dir, "best")
    print(f"Creating soft link {new_link_path} -> {ckpt_name}")
    if osp.exists(new_link_path):
        os.remove(new_link_path)
    os.symlink(ckpt_name, new_link_path, target_is_directory=True)
