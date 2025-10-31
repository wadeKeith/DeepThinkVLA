#!/bin/bash
# This script evaluates DeepThinkVLA checkpoints on LIBERO suites.
# Author: Cheng Yin
# Date: 2025-09
# Copyright (c) Cheng Yin. All rights reserved.
# See LICENSE file in the project root for license information.

set -x

export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

export TOKENIZERS_PARALLELISM=false
export SWANLAB_PROJECT_NAME='deepthinkvla'
export SWANLAB_API_KEY='your_api_key_here'
export SWANLAB_MODE='disabled' # cloud-only, local, disabled
export CUDA_VISIBLE_DEVICES=0


# libero_object, libero_spatial, libero_goal, libero_10, libero_90

python -m experiments.run_libero_eval \
    --pretrained_checkpoint yinchenghust/deepthinkvla_libero_cot_sft \
    --num_images_in_input 2 \
    --task_suite_name libero_10 \
    --max_new_tokens 2048 \
    --project_name $SWANLAB_PROJECT_NAME \
    --swanlab_api_key $SWANLAB_API_KEY \
    --swanlab_mode $SWANLAB_MODE \
    --seed 429 \
    --panel_width_px 812 \
