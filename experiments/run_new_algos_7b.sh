#!/usr/bin/env bash
# Serial 7B training of the 6 new-algorithm runs on dapo-math (unified setting,
# identical to baseline/ar/difffilter: Qwen2.5-7B, GRPO, 300 steps, 8 GPU, rollout 2048).
# Each run -> df_ckpts_7b/<name>; ray cleaned between runs.
set -uo pipefail
ROOT=/apdcephfs_zwfy14/share_304380933/aldenliang
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR=$ROOT/df_logs_7b; mkdir -p "$LOGDIR"
export TOTAL_STEPS=300 SAVE_FREQ=100 CKPT_ROOT=$ROOT/df_ckpts_7b
DATA=$ROOT/data/dapo_math

DFsync="trainer.v1.trainer_mode=dataflex_sync"
declare -A RUN
RUN[gfpo]="$DFsync +dataflex.mechanism=select +dataflex.scorer.name=reward_difficulty +dataflex.actuator.name=gfpo +dataflex.actuator.params.k=3 +dataflex.actuator.params.metric=efficiency +dataflex.warmup_step=0"
RUN[maxvar]="$DFsync +dataflex.mechanism=select +dataflex.scorer.name=reward_difficulty +dataflex.actuator.name=max_variance +dataflex.actuator.params.keep_fraction=0.5 +dataflex.warmup_step=0"
RUN[topk]="$DFsync +dataflex.mechanism=select +dataflex.scorer.name=advantage_magnitude +dataflex.actuator.name=topk_fraction +dataflex.actuator.params.fraction=0.5 +dataflex.warmup_step=0"
RUN[per]="$DFsync +dataflex.mechanism=reweight +dataflex.scorer.name=advantage_magnitude +dataflex.actuator.name=per_advantage +dataflex.actuator.params.alpha=0.5 +dataflex.warmup_step=0"
RUN[softmax]="$DFsync +dataflex.mechanism=reweight +dataflex.scorer.name=advantage_magnitude +dataflex.actuator.name=softmax +dataflex.actuator.params.temperature=1.0 +dataflex.warmup_step=0"
RUN[diffband]="$DFsync +dataflex.mechanism=reweight +dataflex.scorer.name=reward_difficulty +dataflex.actuator.name=difficulty_band +dataflex.warmup_step=0"

ORDER="${*:-gfpo maxvar topk per softmax diffband}"
for name in $ORDER; do
  if [ -d "$CKPT_ROOT/$name/global_step_300" ]; then
    echo ">> [$(date +%H:%M:%S)] $name already DONE, skip"; continue
  fi
  echo "=================================================================="
  echo ">> [$(date +%H:%M:%S)] START $name"
  echo ">> DF_ARGS=${RUN[$name]}"
  echo "=================================================================="
  EXP_NAME="$name" DATA_DIR="$DATA" DF_ARGS="${RUN[$name]}" \
    bash "$HERE/train_one_7b.sh" > "$LOGDIR/$name.log" 2>&1
  echo ">> [$(date +%H:%M:%S)] END $name (exit $?)"
  ray stop --force >/dev/null 2>&1 || true
  pkill -9 -f raylet 2>/dev/null || true; pkill -9 -f "ray::" 2>/dev/null || true
  sleep 10
done
echo ">> [$(date +%H:%M:%S)] ALL NEW RUNS DONE"
EOF
