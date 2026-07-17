#!/bin/bash

# ----------- contidition step 10, sliding_window_size 100, lambda_v 0.01
python3 train.py --train_dir ../data/train_1min --test_dir ../data/test_1min \
                 --output_dir checkpoints/layers2_win100_schedule100_condition10_detach \
                 --batch_size 32 --seq_len 900 --max_seq_len 4500 \
                 --frame_emb_size 200 --d_pose_vec 50 --pose_emb_size 50 \
                 --d_inner 1024 --n_layers 2 \
                 --sliding_windown_size 100 --condition_step 10 --lambda_v 0.01

# ----------- contidition step 10, sliding_window_size 500, lambda_v 0.01
python3 train.py --train_dir ../data/train_1min --test_dir ../data/test_1min \
                 --output_dir checkpoints/layers2_win500_schedule100_condition10_detach \
                 --batch_size 32 --seq_len 900 --max_seq_len 4500 \
                 --frame_emb_size 200 --d_pose_vec 50 --pose_emb_size 50 \
                 --d_inner 1024 --n_layers 2 \
                 --sliding_windown_size 500 --condition_step 10 --lambda_v 0.01

# ----------- contidition step 10, sliding_window_size 900, lambda_v 0.01
python3 train.py --train_dir ../data/train_1min --test_dir ../data/test_1min \
                 --output_dir checkpoints/layers2_win900_schedule100_condition10_detach \
                 --batch_size 32 --seq_len 900 --max_seq_len 4500 \
                 --frame_emb_size 200 --d_pose_vec 50 --pose_emb_size 50 \
                 --d_inner 1024 --n_layers 2 \
                 --sliding_windown_size 900 --condition_step 10 --lambda_v 0.01

