
$save_dir = "/home/lyy/workspace/zero-shot-GLS/tmp_saves/LS_CNN/RNN"
# $save_dir = "/home/lyy/workspace/zero-shot-GLS/tmp_saves/LS_CNN/RNN"
$gen_path = "/home/lyy/workspace/zero-shot-GLS/datasets/imdb/RNN/full/imdb_s2_b3.csv"
$gt_path = "/home/lyy/workspace/zero-shot-GLS/datasets/imdb/imdb.csv"
# $gen_path = "/home/lyy/workspace/zero-shot-GLS/datasets/imdb/imdb_s2_c2_t0.005_b5.csv"
# $gt_path = "/home/lyy/workspace/zero-shot-GLS/datasets/imdb/imdb.csv"

# python main.py \
#     -save-dir /home/lyy/workspace/zero-shot-GLS/tmp_saves/LS_CNN \
#     -gen-path /home/lyy/workspace/zero-shot-GLS/datasets/imdb/imdb_s2_c2_t0.005_b5.csv \
#     -gt-path /home/lyy/workspace/zero-shot-GLS/datasets/imdb/imdb.csv \
#     -idx-gpu 1


python main.py \
    -save-dir $save_dir\
    -gen-path $gen_path\
    -gt-path $gt_path\
    -idx-gpu 1


python main.py \
    -save-dir /home/lyy/workspace/zero-shot-GLS/tmp_saves/LS_CNN/RNN \
    -save-ckp /home/lyy/workspace/zero-shot-GLS/tmp_saves/LS_CNN/RNN/best_epochs.pt\
    -gen-path /home/lyy/workspace/zero-shot-GLS/datasets/imdb/RNN/full/imdb_s2_b3.csv \
    -gt-path /home/lyy/workspace/zero-shot-GLS/datasets/imdb/imdb.csv \
    -test
