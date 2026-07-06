#!/usr/bin/env bash
# Mixer round driver — jizhicfs / H20. Runs the dataflex_mix_sync trainer (custom sampler)
# on the 3-domain set (math/logic/science). Matched to the main-campaign 7B setting
# (flash_attn, 8192 response, 0.85 mem, multidomain reward). Only the MIXER varies via $DF_ARGS.
#
# Two pieces (see examples/run_mix_grpo.sh):
#   trainer.v1.trainer_mode=dataflex_mix_sync
#   trainer.v1.sampler.custom_sampler.path=pkg://dataflex_verl.replay_buffer
set -xeuo pipefail

ROOT=/jizhicfs/aldenliang
export PATH=$ROOT/miniconda3/envs/verl/bin:$PATH
MODEL=${MODEL:-$ROOT/models/Qwen2.5-7B-Instruct}
DATA=${DATA_DIR:-$ROOT/data/multidomain_3}
CKPT_ROOT=${CKPT_ROOT:-$ROOT/df_ckpts_mix_seeds}
EXP_NAME=${EXP_NAME:?set EXP_NAME}
DF_ARGS=${DF_ARGS:?set DF_ARGS}     # the +dataflex.* mixer block
TOTAL_STEPS=${TOTAL_STEPS:-300}
SAVE_FREQ=${SAVE_FREQ:-100}
SEED=${SEED:-1}

export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export RAY_USE_MULTIPROCESSING_CPU_COUNT=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1

# shellcheck disable=SC2086
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA/train.parquet" \
    data.val_files="$DATA/test.parquet" \
    custom_reward_function.path="$ROOT/DataFlex-RL/src/dataflex_verl/rewards/multidomain_reward.py" \
    custom_reward_function.name=compute_score \
    reward.custom_reward_function.path="$ROOT/DataFlex-RL/src/dataflex_verl/rewards/multidomain_reward.py" \
    reward.custom_reward_function.name=compute_score \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.dataloader_num_workers=0 \
    data.seed=$SEED \
    actor_rollout_ref.model.path="$MODEL" \
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    ray_kwargs.ray_init.num_cpus=64 \
    trainer.v1.trainer_mode=dataflex_mix_sync \
    trainer.v1.sampler.custom_sampler.path=pkg://dataflex_verl.replay_buffer \
    trainer.v1.sampler.custom_sampler.name=DataFlexMixReplayBuffer \
    +dataflex.mechanism=mix \
    "+dataflex.domains=[math,logic,science]" \
    +dataflex.domain_key=domain \
    $DF_ARGS \
    trainer.logger=console \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=50 \
    trainer.default_local_dir="$CKPT_ROOT/$EXP_NAME" \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.total_epochs=100 \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.project_name=dataflex_compare_mix \
    trainer.experiment_name="$EXP_NAME"
