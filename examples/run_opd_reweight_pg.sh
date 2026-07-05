#!/usr/bin/env bash
# DataFlex-verl + On-Policy Distillation (OPD) — REWEIGHT example.
#
# Teacher-student divergence as a token-level reweighting signal. verl runs PG OPD
# (use_policy_gradient=true) so the distillation update goes through the PPO
# policy-gradient path, which honors rollout_is_weights — DataFlex's per-token
# weights are multiplied in with ZERO change to verl.
#
#   scorer   distill_kl   (k_t = student_logp - teacher_logp, |k| by default)
#   actuator advantage_reweight (damps tokens; here keyed on the distill signal)
#
# The teacher shares the student tokenizer/vocab (same model family). Enable OPD:
# a dedicated teacher pool scores each rollout, producing teacher_logprobs.
#
# NOTE: reweight/select REQUIRE PG OPD. Under GKD (use_policy_gradient=false) verl
# backprops the distillation loss directly and ignores rollout_is_weights, so the
# reweighting would silently no-op — DataFlex rejects that combo at mount time.
set -xeuo pipefail

ROOT=/apdcephfs_zwfy14/share_304380933/aldenliang
MODEL=$ROOT/models/Qwen2.5-7B-Instruct          # student
TEACHER=$ROOT/models/Qwen2.5-14B-Instruct        # teacher (same family/vocab)
DATA=$ROOT/data/dapo_math

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
    distillation.n_gpus_per_node=2 \
    distillation.nnodes=1 \
    distillation.teacher_models.teacher_model.model_path="$TEACHER" \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.8 \
    distillation.distillation_loss.loss_mode=k3 \
    distillation.distillation_loss.use_policy_gradient=true \
    distillation.distillation_loss.policy_loss_mode=vanilla \
    distillation.distillation_loss.clip_ratio_low=0.2 \
    distillation.distillation_loss.clip_ratio_high=0.28 \
    distillation.distillation_loss.use_task_rewards=true \
    distillation.distillation_loss.distillation_loss_coef=1.0 \
    trainer.v1.trainer_mode=dataflex_sync \
    +dataflex.mechanism=reweight \
    +dataflex.scorer.name=distill_kl \
    +dataflex.scorer.params.mode=abs \
    +dataflex.actuator.name=advantage_reweight \
    +dataflex.actuator.params.alpha=0.5 \
    +dataflex.warmup_step=0 \
    trainer.logger=console \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=5 \
    trainer.project_name=dataflex_opd \
    trainer.experiment_name=opd_reweight_pg
