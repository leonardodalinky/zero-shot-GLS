"""Dataset from .csv file with single specified column."""
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    def __init__(self, csv_path: str | Path, col_name: str = "plaintext"):
        super().__init__()
        df = pd.read_csv(csv_path)
        assert col_name in df.columns, f"Column {col_name} not in {csv_path}"
        self.data = df[col_name].to_list()
        self.data = [str(x) for x in self.data]

        self.col_name = col_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "text": self.data[idx],
        }
