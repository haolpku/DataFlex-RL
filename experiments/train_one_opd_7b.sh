#!/usr/bin/env bash
# 7B OPD training driver (RL + on-policy distillation). Companion to train_one_7b.sh,
# but places the teacher-pool / GPU-carve-out / KL switches AFTER $DF_ARGS is not
# enough — so all OPD-critical overrides live here at the END where hydra's
# last-value-wins guarantees they take effect (train_one_7b.sh hardcodes
# n_gpus_per_node=8 and use_kl_loss=True after its $DF_ARGS slot).
#
# Only $DF_ARGS (the DataFlex + distillation block) varies per run; see run_opd_7b.sh.
set -xeuo pipefail

ROOT=/apdcephfs_zwfy14/share_304380933/aldenliang
MODEL=$ROOT/models/Qwen2.5-7B-Instruct
DATA=${DATA_DIR:-$ROOT/data/dapo_math}
CKPT_ROOT=${CKPT_ROOT:-$ROOT/df_ckpts_7b_opd}
EXP_NAME=${EXP_NAME:?set EXP_NAME}
DF_ARGS=${DF_ARGS:-}
TOTAL_STEPS=${TOTAL_STEPS:-300}
SAVE_FREQ=${SAVE_FREQ:-100}
SEED=${SEED:-1}
STUDENT_GPUS=${STUDENT_GPUS:-6}     # student trainer GPUs; teacher pool takes the rest

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
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.dataloader_num_workers=0 \
    data.seed=$SEED \
    actor_rollout_ref.model.path="$MODEL" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    ray_kwargs.ray_init.num_cpus=64 \
    $DF_ARGS \
    trainer.logger=console \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$STUDENT_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=50 \
    trainer.default_local_dir="$CKPT_ROOT/$EXP_NAME" \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.total_epochs=100 \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.project_name=dataflex_opd_7b \
    trainer.experiment_name="$EXP_NAME"
