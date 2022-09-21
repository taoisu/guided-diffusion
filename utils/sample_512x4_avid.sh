#!/bin/bash

set -xe

cd "$(dirname $0)/.."

# prepare images
DATA_DIR="data/Random"
BASE_SAMPLES="datasets/128_samples.npz"
python3 scripts/super_res_prep.py --data_dir $DATA_DIR --out_path $BASE_SAMPLES --num_samples 8 --large_size 512 --small_size 128

# sample diffusion model
export CUDA_VISIBLE_DEVICES=1
MODEL_DIR="/tmp/openai-2022-09-19-21-22-40-071452"
MODEL_FLAGS="--attention_resolutions 32,16,8 --large_size 512 --small_size 128 --learn_sigma True --num_channels 160 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True --use_fp16 True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
SAMPLE_FLAGS="--batch_size 4 --use_ddim True --timestep_respacing 250 --num_samples 8"
MODEL_PATH="$MODEL_DIR/ema_0.9999_040000.pt"
python3 scripts/super_res_sample.py --base_samples $BASE_SAMPLES --model_path $MODEL_PATH $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS

# save images
OUT_DIR="data/tmp"
NPZ_PATH="/tmp/samples_8x512x512x3.npz"
python3 scripts/super_res_npz2img.py --npz_path $NPZ_PATH --out_dir $OUT_DIR