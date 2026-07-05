#!/usr/bin/env bash
# OPD comparison runs (M3/M4) — RUN AFTER the current 6-run pipeline + eval finish.
# Do NOT launch while run_new_algos_7b.sh is active: OPD adds a teacher pool and will
# contend for GPUs. This script only defines the runs; start it manually when free.
#
# PREREQUISITE: editable install must point at the branch with distill_kl/distill_gap
# + validate_opd_compat + the generalized mix trainer (feat/opd-fusion merged into the
# installed DataFlex-RL). Verify:
#   python -c "from dataflex_verl.core.registry import REGISTRY; import dataflex_verl.scorers; \
#     print('distill_kl' in REGISTRY.list('scorer'))"
#
# GPU CARVE-OUT (must tune to the box): OPD needs a teacher pool of
# n_gpus_per_node*nnodes GPUs SEPARATE from the student's trainer.n_gpus_per_node.
# The defaults below assume 8 GPUs total: 6 student + 2 teacher. Adjust
# trainer.n_gpus_per_node / distillation.n_gpus_per_node so they sum to the box.
#
# Unified setting otherwise identical to the RL runs (Qwen2.5-7B, GRPO, dapo-math,
# 300 steps, rollout 2048) so OPD variants are directly comparable to baseline/RL.
set -uo pipefail
ROOT=/apdcephfs_zwfy14/share_304380933/aldenliang
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR=$ROOT/df_logs_7b_opd; mkdir -p "$LOGDIR"
export TOTAL_STEPS=300 SAVE_FREQ=100 CKPT_ROOT=$ROOT/df_ckpts_7b_opd
DATA=$ROOT/data/dapo_math
TEACHER=${TEACHER:-$ROOT/models/Qwen2.5-14B-Instruct}   # same family/vocab as 7B student

# Teacher pool: single teacher, PG OPD (k3), task rewards on (RL + OPD mixed loss).
# 6 student GPUs + 2 teacher GPUs. Trainer GPUs overridden per-run below.
OPD_BASE="distillation.enabled=true \
distillation.n_gpus_per_node=2 distillation.nnodes=1 \
distillation.teacher_models.teacher_model.model_path=$TEACHER \
distillation.teacher_models.teacher_model.inference.name=vllm \
distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.8 \
distillation.distillation_loss.loss_mode=k3 \
distillation.distillation_loss.use_policy_gradient=true \
distillation.distillation_loss.policy_loss_mode=vanilla \
distillation.distillation_loss.clip_ratio_low=0.2 \
distillation.distillation_loss.clip_ratio_high=0.28 \
distillation.distillation_loss.use_task_rewards=true \
distillation.distillation_loss.distillation_loss_coef=1.0 \
actor_rollout_ref.actor.use_kl_loss=False algorithm.use_kl_in_reward=False"

DFsync="trainer.v1.trainer_mode=dataflex_sync"
declare -A RUN
# Pure PG OPD (no DataFlex scheduling) — the OPD baseline to beat.
RUN[opd_base]="$OPD_BASE $DFsync"
# OPD + DataFlex reweight on teacher divergence (token-level focus).
RUN[opd_reweight]="$OPD_BASE $DFsync +dataflex.mechanism=reweight +dataflex.scorer.name=distill_kl +dataflex.scorer.params.mode=abs +dataflex.actuator.name=advantage_reweight +dataflex.actuator.params.alpha=0.5 +dataflex.warmup_step=0"
# OPD + DataFlex select: keep high-divergence samples (focus the mixed update).
RUN[opd_select]="$OPD_BASE $DFsync +dataflex.mechanism=select +dataflex.scorer.name=distill_gap +dataflex.scorer.params.mode=abs +dataflex.actuator.name=topk_fraction +dataflex.actuator.params.fraction=0.5 +dataflex.actuator.params.largest=True +dataflex.warmup_step=0"

ORDER="${*:-opd_base opd_reweight opd_select}"
for name in $ORDER; do
  if [ -d "$CKPT_ROOT/$name/global_step_300" ]; then
    echo ">> [$(date +%H:%M:%S)] $name already DONE, skip"; continue
  fi
  echo "=================================================================="
  echo ">> [$(date +%H:%M:%S)] START $name"
  echo ">> DF_ARGS=${RUN[$name]}"
  echo "=================================================================="
  EXP_NAME="$name" DATA_DIR="$DATA" DF_ARGS="${RUN[$name]}" \
    bash "$HERE/train_one_opd_7b.sh" > "$LOGDIR/$name.log" 2>&1
  echo ">> [$(date +%H:%M:%S)] END $name (exit $?)"
  ray stop --force >/dev/null 2>&1 || true
  pkill -9 -f raylet 2>/dev/null || true; pkill -9 -f "ray::" 2>/dev/null || true
  sleep 10
done
echo ">> [$(date +%H:%M:%S)] ALL OPD RUNS DONE"
