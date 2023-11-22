"""Dataset from .csv file with single specified column."""
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    def __init__(self, data: list[str]):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "text": self.data[idx],
        }


def create_train_test_datasets(
    csv_path: str | Path,
    col_name: str = "plaintext",
    train_ratio: float = 0.85,
    seed: int = 42,
) -> tuple[CSVDataset, CSVDataset]:
    df = pd.read_csv(csv_path)
    assert col_name in df.columns, f"Column {col_name} not in {csv_path}"
    train = df.sample(frac=train_ratio, random_state=seed)
    train_data: list[str] = train[col_name].to_list()
    train_data = [str(x) for x in train_data]
    test = df.drop(train.index)
    test_data: list[str] = test[col_name].to_list()
    test_data = [str(x) for x in test_data]
    return CSVDataset(train_data), CSVDataset(test_data)
