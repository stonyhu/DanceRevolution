#! /bin/bash

# export CUDA_VISIBLE_DEVICES=3

model_dir=checkpoints_windowsize100_hidden1024_fixedsteps10_alpha0.01

python train.py --train_dir data/train \
                --output_dir ${model_dir} \
                --batch_size 32 \
                --lr 0.0001 \
                --dropout 0.05 \
                --frame_dim 438 \
                --encoder_hidden_size 1024 \
                --pose_dim 50 \
                --decoder_hidden_size 512 \
                --seq_len 900 \
                --max_seq_len 4500 \
                --num_heads 8 \
                --num_layers 3 \
                --window_size 100 \
                --fixed_step 10 \
                --alpha 0.01 \
                --save_per_epochs 500
