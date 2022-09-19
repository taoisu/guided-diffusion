#!/bin/bash

set -xe

cd "$(dirname $0)/.."

MODEL_FLAGS="--attention_resolutions 32,16,8 --large_size 512 --small_size 128 --learn_sigma True --num_channels 160 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 5e-5 --batch_size 4"
DATA_DIR="data/Random"

mpiexec -n 2 python3 scripts/super_res_train.py --data_dir $DATA_DIR $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS