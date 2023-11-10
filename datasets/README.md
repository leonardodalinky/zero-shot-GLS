# Datasets

This directory only contains the script to produce the datasets. The datasets themselves are not included in this repository.

However, you should link the datasets to this directory.

Currently, we use the following datasets:
* IDMB
* Twitter

The structure of the `datasets/` should be as follows:
```
datasets
├── imdb -> /PATH/TO/LINK
│   └── imdb.csv
├── imdb_gen.ipynb
├── README.md
├── twitter -> /PATH/TO/LINK
│   └── twitter.csv
└── twitter_gen.ipynb
```

## Data format

We utilize `.csv` format to store the datasets. The `.csv` files should have the following columns:
* `sentence_id`: The id of the sentence.
* `plaintext`: The plaintext of the sentence.
* Stage 1:
  * `enc_bits`: The encoded bits of the sentence.
* Stage 2:
  * `stegotext`: The stegotext of the `enc_bits`.
  * `enc_seed`: The seed used to generate the stegotext.
* Stage 3:
  * `dec_bits`: The decoded bits of the `stegotext`.
* Stage 4:
  * `dec_plaintext`: The decoded plaintext of the `dec_bits`.
