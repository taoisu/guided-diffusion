#!/bin/bash

set -xe

cd "$(dirname $0)/.."

MODEL_FLAGS="--attention_resolutions 32,16,8 --large_size 512 --small_size 128 --learn_sigma True --num_channels 192 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True --use_fp16 True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
SAMPLE_FLAGS="--batch_size 4 --use_ddim True --timestep_respacing 250"
BASE_SAMPLES="datasets/128_samples.npz"
MODEL_PATH="/tmp/openai-2022-09-18-21-24-48-880279/ema_0.9999_001000.pt"

export CUDA_VISIBLE_DEVICES=1
python3 scripts/super_res_sample.py --base_samples $BASE_SAMPLES --model_path $MODEL_PATH $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS