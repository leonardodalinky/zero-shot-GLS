# Zero-shot Generative Linguistic Steganography

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
4. Create a temp folder in HDD for storing and sharing files:
```shell
# In our server only.
mkdir -p /media/data1/share/zgls
ln -s /media/data1/share/zgls tmp_saves
```
