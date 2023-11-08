# Stego Pipeline

## Setup

First, download all models.
```shell
python download_models.py
```

P.S.: If you set `tmp_saves/` correctly, you can skip this step.

## Stages

* stage1 (s1): Plaintext to enc_bits.
* stage2 (s2): Enc_bits to stegotext.
* stage3 (s3): Stegotext to dec_bits.
* stage4 (s4): Dec_bits to dec_plaintext.

## Stego modes

For now, we have 3 modes:
* `--mode=cover`: mimic the cover text.
