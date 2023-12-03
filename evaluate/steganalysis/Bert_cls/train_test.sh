gen_path_zgls="zero-shot-GLS/datasets/imdb/imdb_s2_c2_t0.005_b5.csv"
save_dir_zgls="zero-shot-GLS/tmp_saves/bert/zgls"
save_ckpt_zgls="zero-shot-GLS/tmp_saves/bert/zgls/best_epochs.pt"

gen_path_ADG="zero-shot-GLS/datasets/imdb/ADG/full/imdb_s2.csv"
save_dir_ADG="zero-shot-GLS/tmp_saves/bert/ADG"
save_ckpt_ADG="zero-shot-GLS/tmp_saves/bert/ADG/best_epochs.pt"

gen_path_NLS="zero-shot-GLS/datasets/imdb/NLS/imdb_s2_b3.csv"
save_dir_NLS="zero-shot-GLS/tmp_saves/bert/NLS"
save_ckpt_NLS="zero-shot-GLS/tmp_saves/bert/NLS/best_epochs.pt"

gen_path_RNN="zero-shot-GLS/datasets/imdb/RNN/full/imdb_s2_b3.csv"
save_dir_RNN="zero-shot-GLS/tmp_saves/bert/RNN"
save_ckpt_RNN="zero-shot-GLS/tmp_saves/bert/RNN/best_epochs.pt"

gen_path_SAAC="zero-shot-GLS/datasets/imdb/SAAC/imdb_s2_k50_d1.00.csv"
save_dir_SAAC="zero-shot-GLS/tmp_saves/bert/SAAC"
save_ckpt_SAAC="zero-shot-GLS/tmp_saves/bert/SAAC/best_epochs.pt"

gen_path_VAE="zero-shot-GLS/datasets/imdb/VAE/full/imdb_s2_b3.csv"
save_dir_VAE="zero-shot-GLS/tmp_saves/bert/VAE"
save_ckpt_VAE="zero-shot-GLS/tmp_saves/bert/VAE/best_epochs.pt"

gt_path="zero-shot-GLS/datasets/imdb/imdb.csv"

export CUDA_VISIBLE_DEVICES="1"
# ---------------RNN--------------
python main.py \
    -save-dir $save_dir_RNN\
    -gen-path $gen_path_RNN\
    -gt-path $gt_path

# ---------------zgls--------------
python main.py \
    -save-dir $save_dir_zgls\
    -gen-path $gen_path_zgls\
    -gt-path $gt_path

# ---------------NLS--------------
python main.py \
    -save-dir $save_dir_NLS\
    -gen-path $gen_path_NLS\
    -gt-path $gt_path

# ---------------VAE--------------
python main.py \
    -save-dir $save_dir_VAE\
    -gen-path $gen_path_VAE\
    -gt-path $gt_path

# ---------------SAAC--------------
python main.py \
    -save-dir $save_dir_SAAC\
    -gen-path $gen_path_SAAC\
    -gt-path $gt_path
