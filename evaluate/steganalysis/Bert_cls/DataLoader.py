import random
from pathlib import Path

import pandas
import sklearn.model_selection as ms
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class BertDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "label": torch.tensor(self.label[idx], dtype=torch.long),  # 'positive' or 'negative
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    @classmethod
    def load_data(cls, args):
        gt_path = Path(args.gt_path)
        generated_path = Path(args.gen_path)
        # Load your data from the specified path
        # and return a list of text samples
        gt = pandas.read_csv(gt_path)["plaintext"].tolist()
        df = pandas.read_csv(generated_path)["stegotext"].tolist()

        sample_size = 4000
        # random.seed(2024)
        gt = random.sample(gt, sample_size)

        # Split gt and df into training and test set
        train_cover, test_cover, train_stego, test_stego = ms.train_test_split(
            gt, df, test_size=0.25, random_state=2024
        )
        train_label = [0] * len(train_cover) + [1] * len(train_stego)
        test_label = [0] * len(test_cover) + [1] * len(test_stego)
        return cls(train_cover + train_stego, train_label), cls(test_cover + test_stego, test_label)
