#!/usr/bin/env bash
# DataFlex-verl + On-Policy Distillation — MIX example (MOPD + domain mixing).
#
# Multi-teacher OPD (MOPD): one domain-specialized teacher per domain, routed by
# data_source (distillation.teacher_key). DataFlex Mixer steers how much each domain
# is sampled. Mix runs at the replay-buffer / pre-rollout layer, so it does NOT go
# through the loss — it works under BOTH PG and GKD OPD (unlike reweight/select).
#
#   trainer_mode dataflex_mix_sync + DataFlexMixReplayBuffer
#   scorer   reward_difficulty     (see NOTE)
#   actuator reward_gap            (favor the lagging domain)
#
# NOTE (honest limitation): DataFlexMixSyncTrainer currently accumulates each domain's
# mean *reward* to drive proportions. Mixing by the teacher-student *divergence*
# (distill_gap per domain) needs the mix trainer to feed the teacher signal into the
# DomainStatsTracker — that hook is planned in research/05 (M3/M4). Until then this
# example combines MOPD training with reward-driven mixing (still a valid, useful combo).
set -xeuo pipefail

ROOT=/apdcephfs_zwfy14/share_304380933/aldenliang
MODEL=$ROOT/models/Qwen2.5-7B-Instruct
TEACHER_MATH=$ROOT/models/Qwen2.5-Math-7B-Instruct     # math-specialized teacher
TEACHER_CODE=$ROOT/models/Qwen2.5-Coder-7B-Instruct    # code-specialized teacher
DATA=$ROOT/data/math_code_2domain   # multi-domain parquet (data_source per row + domain col)

export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export RAY_USE_MULTIPROCESSING_CPU_COUNT=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA/train.parquet" \
    data.val_files="$DATA/test.parquet" \
    data.shuffle=true \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path="$MODEL" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    algorithm.use_kl_in_reward=False \
    ray_kwargs.ray_init.num_cpus=64 \
    distillation.enabled=true \
    distillation.n_gpus_per_node=4 \
    distillation.nnodes=1 \
    distillation.teacher_key=data_source \
    +distillation.teacher_models.math.key=math \
    +distillation.teacher_models.math.model_path="$TEACHER_MATH" \
    +distillation.teacher_models.math.num_replicas=1 \
    +distillation.teacher_models.math.inference.name=vllm \
    +distillation.teacher_models.math.inference.tensor_model_parallel_size=2 \
    +distillation.teacher_models.code.key=code \
    +distillation.teacher_models.code.model_path="$TEACHER_CODE" \
    +distillation.teacher_models.code.num_replicas=1 \
    +distillation.teacher_models.code.inference.name=vllm \
    +distillation.teacher_models.code.inference.tensor_model_parallel_size=2 \
    distillation.distillation_loss.loss_mode=k3 \
    distillation.distillation_loss.use_policy_gradient=true \
    distillation.distillation_loss.policy_loss_mode=vanilla \
    distillation.distillation_loss.use_task_rewards=true \
    trainer.v1.trainer_mode=dataflex_mix_sync \
    trainer.v1.sampler.custom_sampler.path=pkg://dataflex_verl.replay_buffer \
    trainer.v1.sampler.custom_sampler.name=DataFlexMixReplayBuffer \
    +dataflex.mechanism=mix \
    "+dataflex.domains=[math,code]" \
    +dataflex.domain_key=domain \
    +dataflex.scorer.name=reward_difficulty \
    +dataflex.actuator.name=reward_gap \
    +dataflex.actuator.params.temperature=1.0 \
    +dataflex.actuator.params.floor=0.05 \
    +dataflex.warmup_step=5 \
    +dataflex.update_step=5 \
    +dataflex.window=50 \
    trainer.logger=console \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=10 \
    trainer.project_name=dataflex_opd \
    trainer.experiment_name=opd_mix_mopd
