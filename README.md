# Zero-shot Generative Linguistic Steganography

[![arXiv](https://img.shields.io/badge/arXiv-2403.10856-brightgreen.svg)](https://arxiv.org/abs/2403.10856)
[![star badge](https://img.shields.io/github/stars/leonardodalinky/zero-shot-GLS?style=social)](https://github.com/leonardodalinky/zero-shot-GLS)

This repo is the official implementation of "Zero-shot Generative Linguistic Steganography".

## Setup

1. Create Conda environment and install the requirements:
```shell
conda create -n zgls python=3.10
conda activate zgls
#
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu121/
```
2. Compile external Python module:
```shell
pip install external_tools/zgls-utils
```
3. Install pre-commit hooks:
```shell
pre-commit install
```
4. Create a temp folder in HDD for storing and sharing files, for example:
```shell
# DO NOT copy directly!
mkdir -p $SHARE_FOLDER/zgls
ln -s $SHARE_FOLDER/zgls tmp_saves
ln -s $SHARE_FOLDER/datasets/imdb datasets/imdb
ln -s $SHARE_FOLDER/datasets/twitter datasets/twitter
```

## Datasets

Check [datasets/](datasets/README.md) section for details.

## Usage

See [scripts/](scripts/README.md) section for details.

## Evaluation

For details of metrics, steganalysis, and language evaluation, check [evaluate/](evaluate/README.md) section.

## Reference

TBD.
