# Zero-shot Generative Linguistic Steganography

[![arXiv](https://img.shields.io/badge/arXiv-2403.10856-brightgreen.svg)](https://arxiv.org/abs/2403.10856)
[![star badge](https://img.shields.io/github/stars/leonardodalinky/zero-shot-GLS?style=social)](https://github.com/leonardodalinky/zero-shot-GLS)

This repo is the official implementation of NAACL'24 paper "[Zero-shot Generative Linguistic Steganography](https://aclanthology.org/2024.naacl-long.289/)".

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

```
@inproceedings{lin2024zgls,
    title = "Zero-shot Generative Linguistic Steganography",
    author = "Lin, Ke  and
      Luo, Yiyang  and
      Zhang, Zijian  and
      Ping, Luo",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.289",
    pages = "5168--5182",
}
```
