#!/bin/bash
# This script launches DeepThinkVLA reinforcement learning with VERL PPO.
# Author: Cheng Yin
# Date: 2025-09
# Copyright (c) Cheng Yin. All rights reserved.
# See LICENSE file in the project root for license information.

set -x

export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

# export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8
# export NCCL_SOCKET_IFNAME=ib0
# export NCCL_IB_DISABLE=0
# export NCCL_NET_GDR_LEVEL=2

export LIBERO_CONFIG_PATH="$(pwd)/src/libero"

export NCCL_DEBUG=WARN 
export SWANLAB_API_KEY='your_swanlab_api_key'
export SWANLAB_LOG_DIR='./logs'
export SWANLAB_MODE='disabled' # cloud-only, local, disabled
export TOKENIZERS_PARALLELISM=true
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export RAY_memory_monitor_refresh_ms=250
export RAY_memory_usage_threshold=0.9
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export RAY_DEBUG_POST_MORTEM=1
# export MIXED_PRECISION=bf16

PROJECT_NAME='deepthinkvla'
EXPERIMENT_NAME='libero_10_deepthinkvla_rl_cot'
SFT_MODEL_PATH="yinchenghust/deepthinkvla_libero_cot_sft"
# DATASET_NAME can be libero_10 (libero_Long), libero_90, libero_spatial, libero_object, libero_goal
DATASET_NAME="libero_10"
NUM_GPUS=8
# If you want to use 2*8 GPU to RL. Set NUM_NODES=2
NUM_NODES=1 
ALIGN_PATH="$(pwd)/scripts/align.json"

###########################################################################################
# verifier.format_coef and verifier.acc_coef can be tuned for better performance.
###########################################################################################
HYDRA_FULL_ERROR=1 python -m verl.trainer.main_ppo \
    data.task_suite_name=$DATASET_NAME \
    data.num_trials_per_task=50 \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.1 \
    data.accuracy_upper_bound=0.9 \
    data.oversample_factor=1 \
    data.train_batch_size=64 \
    data.val_batch_size=496 \
    data.max_prompt_length=672 \
    data.max_response_length=300 \
    data.n_samples=8 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.action_dim_len=7 \
    actor_rollout_ref.model.action_chunks_len=10 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=0 \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_c=3.0 \
    actor_rollout_ref.actor.num_images_in_input=2 \
    actor_rollout_ref.actor.traj_mini_batch_size=8 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.rollout.cot_temperature=1.0 \
    actor_rollout_ref.rollout.action_temperature=1.6 \
    actor_rollout_ref.rollout.experiment_name=$EXPERIMENT_NAME \
    actor_rollout_ref.rollout.micro_batch_size=4 \
    actor_rollout_ref.rollout.val_micro_batch_size=31 \
    actor_rollout_ref.rollout.num_steps_wait=10 \
    actor_rollout_ref.rollout.center_crop=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=4 \
    trainer.test_freq=4 \
    trainer.total_epochs=100 \
    trainer.val_only=False \
    trainer.runtime_env=$ALIGN_PATH \
    trainer.val_before_train=True \
    trainer.update_warmup=0 \
    trainer.max_actor_ckpt_to_keep=40 \
    algorithm.adv_estimator=grpo \
    verifier.acc_coef=1.0 \
    verifier.format_coef=0.4 \
