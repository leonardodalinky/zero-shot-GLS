# Stego Analysis

This module is for stego analysis. It is used to evaluate the performance of steganography algorithms by applying classification methods.

## Requirements

First of all, you must download the pre-trained word-embedding models named as: ("glove.6B.300d.txt")[https://nlp.stanford.edu/data/glove.6B.zip] and unzip it.
```shell
wget https://nlp.stanford.edu/data/glove.6B.zip
```

Then, you should set a soft link to the pre-trained model for `LS_CNN`, `R-BiLSTM-C` and `TS-BiRNN`:
```shell
# Assume you are at steganalysis/
ln -s /path/to/glove.6B.300d.txt LS_CNN/glove_weight/glove.6B.300d.txt
cp LS_CNN/glove_weight/glove.6B/300d.txt R-BiLSTM-C/glove_weight/glove.6B.300d.txt
cp LS_CNN/glove_weight/glove.6B/300d.txt TS-BiRNN/glove_weight/glove.6B.300d.txt
```

Last, you need to install the following packages:
```
pip install torchtext
pip install -U scikit-learn
pip install transformers
```

## How to Run
In each folder, there is a `train_test.sh` file. You need to modify the text path to your target file, then you can run it.

```shell
bash LS_CNN/train_test.sh
```

## File Structure
```
steganalysis
├── Bert_cls
│   ├── Bert_cls.py
│   ├── DataLoader.py
│   ├── main.py
│   ├── train_test.sh
│   └── utils.py
├── LS_CNN
│   ├── DataLoader.py
│   ├── glove_weight
│   │   └── readme
│   ├── LS_CNN.py
│   ├── main.py
│   ├── train.py
│   └── train_test.sh
├── R-BiLSTM-C
│   ├── DataLoader.py
│   ├── glove_weight
│   │   └── readme
│   ├── main.py
│   ├── R_BI_C.py
│   ├── train.py
│   └── train_test.sh
├── README.md
└── TS-BiRNN
    ├── DataLoader.py
    ├── main.py
    ├── train.py
    ├── train_test.sh
    └── TS_BiRNN.py
```
