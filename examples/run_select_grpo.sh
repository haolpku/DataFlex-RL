#!/usr/bin/env bash
# DataFlex-verl SELECT example: GRPO + DAPO-style dynamic sampling.
#
# Drops all-solved / all-failed GRPO groups (keeps groups whose solve rate is
# strictly in (0,1)) by zeroing their loss weight after advantage. Reproduces the
# reward/advantage-driven filtering variant of DAPO.
#
# Zero-config: `pip install dataflex_verl` auto-registers via verl's entry-point
# plugin discovery. verl starts Ray itself — do NOT `ray start` manually.
#
# Verified: Qwen2.5-0.5B / GSM8K / 8x H20, 2 steps, dataflex/kept_frac 0.047->0.078.
set -xeuo pipefail

ROOT=/apdcephfs_zwfy14/share_304380933/aldenliang
MODEL=$ROOT/models/Qwen2.5-0.5B-Instruct
DATA=$ROOT/data/gsm8k

export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export RAY_USE_MULTIPROCESSING_CPU_COUNT=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA/train.parquet" \
    data.val_files="$DATA/test.parquet" \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path="$MODEL" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    ray_kwargs.ray_init.num_cpus=64 \
    trainer.v1.trainer_mode=dataflex_sync \
    +dataflex.mechanism=select \
    +dataflex.scorer.name=group_solve_rate \
    +dataflex.scorer.params.success_threshold=0.5 \
    +dataflex.actuator.name=threshold_band \
    +dataflex.actuator.params.low=0.0 \
    +dataflex.actuator.params.high=1.0 \
    +dataflex.warmup_step=0 \
    trainer.logger=console \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=2 \
    trainer.project_name=dataflex_verl \
    trainer.experiment_name=select_dapo
