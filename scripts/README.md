# Stego Pipeline

## Setup

First, download all models.
```shell
python download_models.py
```

P.S.: If you set `tmp_saves/` correctly, you can skip this step.

## Stages

* stage1 (s1,  encode): Plaintext to enc_bits.
* stage2 (s2, encrypt): Enc_bits to stegotext.
* stage3 (s3, decrypt): Stegotext to dec_bits.
* stage4 (s4,  decode): Dec_bits to dec_plaintext.

## Stego modes

For now, we have 3 modes:
* `--mode=cover`: mimic the cover text.
* TODO
* TODO

## Corpus Hints

For IMDB, we use hint `Movie Reviews`.

For Twitter, we use hint `Twitter`.
