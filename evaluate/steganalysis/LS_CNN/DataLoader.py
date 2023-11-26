import random
import re
from pathlib import Path

import pandas as pd
import sklearn.model_selection as ms
from torchtext import data


class MyData(data.Dataset):
    @staticmethod
    def sort_ket(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, cover=None, stego=None, examples=None, **kwargs):
        """Create an Myself_dataset instance given a path and fields

        Arguments:
                text_field: The field that will be used for text data.
                label_field: The field that will be used for label data.
                path: The path of the data file.
                examples: The examples contain all the data.
                Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        def clean_str(string):
            """
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/
            blob/master/process_data.py
            """
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " 's", string)
            string = re.sub(r"\'ve", " 've", string)
            string = re.sub(r"n\'t", " n't", string)
            string = re.sub(r"\'re", " 're", string)
            string = re.sub(r"\'d", " 'd", string)
            string = re.sub(r"\'ll", " 'll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip()

        self.text_field = text_field
        self.label_field = label_field

        self.text_field.preprocessing = data.Pipeline(clean_str)
        fields = [("text", self.text_field), ("label", self.label_field)]

        if examples is None:
            examples = []
            examples += [data.Example.fromlist([line, "negative"], fields) for line in cover]

            examples += [data.Example.fromlist([line, "positive"], fields) for line in stego]

            # with open(cover_path, 'r', errors='ignore') as f:
            # 	examples += [data.Example.fromlist([line, 'negative'], \
            # 			 	 	 fields) for line in f]

            # with open(stego_path, 'r', errors='ignore') as f:
            # 	examples += [data.Example.fromlist([line, 'positive'], \
            # 				 	 fields) for line in f]

        super(MyData, self).__init__(examples, fields, **kwargs)

    @classmethod
    def split(cls, text_field, label_field, args, state, shuffle=True, **kwargs):
        gt_path = Path(args.gt_path)
        generated_path = Path(args.gen_path)

        # randomly select 4000 record from imdb.csv
        # total_rows = sum(1 for line in open(gt_path)) - 1
        sample_size = 4000
        # skips = sorted(random.sample(range(1, 1 + sample_size), total_rows - sample_size))

        gt = pd.read_csv(gt_path)["plaintext"].tolist()
        random.seed(2024)
        gt = random.sample(gt, sample_size)

        # read 4000 generated text
        df = pd.read_csv(generated_path)["stegotext"].tolist()

        # Split
        train_cover, test_cover, train_stego, test_stego = ms.train_test_split(
            gt, df, test_size=0.25, random_state=2024
        )

        if state == "train":
            print("loading the training data...")
            # cover_path = args.train_cover_dir
            # stego_path = args.train_stego_dir
            examples = cls(text_field, label_field, cover=train_cover, stego=train_stego).examples
            if shuffle:
                random.shuffle(examples)
            val_idx = -500
            return (
                cls(text_field, label_field, examples=examples[:val_idx]),
                cls(text_field, label_field, examples=examples[val_idx:]),
            )

        if state == "test":
            print("loading the testing data...")
            # cover_path = args.test_cover_dir
            # stego_path = args.test_stego_dir
            examples = cls(text_field, label_field, cover=test_cover, stego=test_stego).examples
            return cls(text_field, label_field, examples=examples)
