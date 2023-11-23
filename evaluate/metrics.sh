#!/bin/bash

cd $(dirname $0)

# --------------------------------------
# NOTE: Only modify here!
export CUDA_VISIBLE_DEVICES=0
filepath="../datasets/imdb/SAAC/TODO.csv"
savedir="../tmp_saves/results/imdb/full/TODO"
train_savedir="../tmp_saves/gpt2/TODO"

baseline_modeldir="../tmp_saves/gpt2/imdb/best"
baseline_filepath="../datasets/imdb/imdb.csv"
# --------------------------------------

filename=$(basename $filepath)
filename_no_ext="${filename%.*}"

mkdir -p $savedir/$filename_no_ext

# bpw
echo "Computing bpw..."
python bpw.py $filepath \
    --force \
    -o $savedir/$filename_no_ext/bpw.json
echo ""

# ppl
echo "Computing ppl..."
python ppl.py $filepath \
    --ppl-col "ppl" \
    --force \
    -o $savedir/$filename_no_ext/ppl.json
echo ""

# jsd training
echo "Training model for jsd..."
python GPT/train_gpt.py $filepath \
    --data-col "stegotext" \
    --save-dir $train_savedir/$filename_no_ext
echo ""
# jsd compute
echo "Computing jsd..."
python GPT/jsd.py $baseline_filepath \
    --data-col "plaintext" \
    --force \
    -o $savedir/$filename_no_ext/jsd.json \
    --model-dir1 $baseline_modeldir \
    --model-dir2 $train_savedir/$filename_no_ext/best
echo ""
