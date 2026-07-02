#!/usr/bin/env bash
# DataFlex-verl MIX example: GRPO + dynamic domain proportions.
#
# Buckets prompts by a `domain` column and lets a Mixer steer the sampling
# proportions from each domain's sliding-window mean reward (RewardGapMixer favors
# the lagging domain). Keeps the real `data_source` intact so verl's reward fn works;
# the domain label lives in a separate column (config.dataflex.domain_key, default
# "domain"). Build a multi-domain parquet first — e.g. split GSM8K by question
# length into gsm8k_short / gsm8k_long (see README).
#
# Two pieces work together:
#   trainer.v1.trainer_mode=dataflex_mix_sync        -> tags prompts + updates stats
#   trainer.v1.sampler.custom_sampler.{path,name}    -> DataFlexMixReplayBuffer
# custom_sampler.path MUST use the pkg:// prefix (verl treats bare names as files).
#
# Zero-config plugin: `pip install dataflex_verl` auto-registers via verl's
# entry-point discovery. verl starts Ray itself — do NOT `ray start` manually.
#
# Verified: Qwen2.5-0.5B / gsm8k_2domain / 8x H20, 4 steps, prints
# dataflex/prop_gsm8k_{short,long} (0.5/0.5 in a short smoke — rewards are 0 at cold
# start so there's no gap yet; proportions shift once a domain starts solving).
set -xeuo pipefail

ROOT=/apdcephfs_zwfy14/share_304380933/aldenliang
MODEL=$ROOT/models/Qwen2.5-0.5B-Instruct
DATA=$ROOT/data/gsm8k_2domain   # multi-domain parquet (data_source kept, +domain col)

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
    trainer.v1.trainer_mode=dataflex_mix_sync \
    trainer.v1.sampler.custom_sampler.path=pkg://dataflex_verl.replay_buffer \
    trainer.v1.sampler.custom_sampler.name=DataFlexMixReplayBuffer \
    +dataflex.mechanism=mix \
    "+dataflex.domains=[gsm8k_short,gsm8k_long]" \
    +dataflex.scorer.name=reward_difficulty \
    +dataflex.actuator.name=reward_gap \
    +dataflex.actuator.params.temperature=1.0 \
    +dataflex.actuator.params.floor=0.05 \
    +dataflex.warmup_step=1 \
    +dataflex.update_step=1 \
    +dataflex.window=50 \
    trainer.logger=console \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=4 \
    trainer.project_name=dataflex_verl \
    trainer.experiment_name=mix_2domain
