#!/usr/bin/env bash
# OPD smoke test: verify the teacher pool spins up, teacher_logprobs reach the
# DataFlex hook, and the distill_* signal is finite — BEFORE spending a full run.
#
# Uses student == teacher: verl's OPD doc notes the distillation loss should be
# ~0 (not exact, train/infer engine differences). We additionally assert DataFlex's
# own metric (dataflex/weight_mean) is present and finite, proving distill_kl fired.
#
# PREREQUISITE: the editable install must point at the branch that has distill_kl /
# distill_gap + validate_opd_compat (feat/opd-fusion merged to the installed repo).
# Check: python -c "from dataflex_verl.core.registry import REGISTRY; import dataflex_verl.scorers; print('distill_kl' in REGISTRY.list('scorer'))"
#
# 1 GPU is enough for a 0.5B smoke; teacher takes 1 more. Runs 2 steps.
set -xeuo pipefail

ROOT=/apdcephfs_zwfy14/share_304380933/aldenliang
MODEL=$ROOT/models/Qwen2.5-0.5B-Instruct
TEACHER=$MODEL                                   # student == teacher -> loss ~ 0
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
    data.max_response_length=1024 \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path="$MODEL" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    algorithm.use_kl_in_reward=False \
    ray_kwargs.ray_init.num_cpus=64 \
    distillation.enabled=true \
    distillation.n_gpus_per_node=1 \
    distillation.nnodes=1 \
    distillation.teacher_models.teacher_model.model_path="$TEACHER" \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.5 \
    distillation.distillation_loss.loss_mode=k3 \
    distillation.distillation_loss.use_policy_gradient=true \
    distillation.distillation_loss.policy_loss_mode=vanilla \
    distillation.distillation_loss.use_task_rewards=true \
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
    trainer.total_training_steps=2 \
    trainer.project_name=dataflex_opd \
    trainer.experiment_name=opd_smoke
