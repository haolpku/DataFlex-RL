#!/usr/bin/env bash
# Multi-seed 7B runs for Paper 2 (the benchmark's statistical backbone).
#
# Re-runs all 9 unified-setting configs (baseline + AR + difffilter + the 6 new
# algorithms) across seeds so the paper can report mean ± std, not single-seed points.
# seed 1 already exists in df_ckpts_7b/<name>; this fills seeds 2,3 into
# df_ckpts_7b_seeds/<name>_s<seed>. Idempotent: skips any run whose global_step_300
# checkpoint already exists, so it is safe to re-launch / resume.
#
# Serial on one 8-GPU box (same setting as the originals). For multi-box parallelism,
# the run list this generates is the queue a central dispatcher hands out (see the
# distributed plan); each item is one (name, seed) job runnable via train_one_7b.sh.
set -uo pipefail
ROOT=/apdcephfs_zwfy14/share_304380933/aldenliang
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR=$ROOT/df_logs_7b_seeds; mkdir -p "$LOGDIR"
export TOTAL_STEPS=300 SAVE_FREQ=100 CKPT_ROOT=$ROOT/df_ckpts_7b_seeds
DATA=$ROOT/data/dapo_math

DFsync="trainer.v1.trainer_mode=dataflex_sync"
declare -A RUN
RUN[baseline]=""
RUN[ar]="$DFsync +dataflex.mechanism=reweight +dataflex.scorer.name=token_prob +dataflex.actuator.name=advantage_reweight +dataflex.actuator.params.alpha=0.5 +dataflex.warmup_step=0"
RUN[difffilter]="$DFsync +dataflex.mechanism=select +dataflex.scorer.name=group_solve_rate +dataflex.scorer.params.success_threshold=0.5 +dataflex.actuator.name=threshold_band +dataflex.actuator.params.low=0.2 +dataflex.actuator.params.high=0.8 +dataflex.warmup_step=0"
RUN[gfpo]="$DFsync +dataflex.mechanism=select +dataflex.scorer.name=reward_difficulty +dataflex.actuator.name=gfpo +dataflex.actuator.params.k=3 +dataflex.actuator.params.metric=efficiency +dataflex.warmup_step=0"
RUN[maxvar]="$DFsync +dataflex.mechanism=select +dataflex.scorer.name=reward_difficulty +dataflex.actuator.name=max_variance +dataflex.actuator.params.keep_fraction=0.5 +dataflex.warmup_step=0"
RUN[topk]="$DFsync +dataflex.mechanism=select +dataflex.scorer.name=advantage_magnitude +dataflex.actuator.name=topk_fraction +dataflex.actuator.params.fraction=0.5 +dataflex.warmup_step=0"
RUN[per]="$DFsync +dataflex.mechanism=reweight +dataflex.scorer.name=advantage_magnitude +dataflex.actuator.name=per_advantage +dataflex.actuator.params.alpha=0.5 +dataflex.warmup_step=0"
RUN[softmax]="$DFsync +dataflex.mechanism=reweight +dataflex.scorer.name=advantage_magnitude +dataflex.actuator.name=softmax +dataflex.actuator.params.temperature=1.0 +dataflex.warmup_step=0"
RUN[diffband]="$DFsync +dataflex.mechanism=reweight +dataflex.scorer.name=reward_difficulty +dataflex.actuator.name=difficulty_band +dataflex.warmup_step=0"

ALGOS="${ALGOS:-baseline ar difffilter gfpo maxvar topk per softmax diffband}"
SEEDS="${SEEDS:-2 3}"

for seed in $SEEDS; do
  for name in $ALGOS; do
    exp="${name}_s${seed}"
    if [ -d "$CKPT_ROOT/$exp/global_step_300" ]; then
      echo ">> [$(date +%H:%M:%S)] $exp already DONE, skip"; continue
    fi
    echo "=================================================================="
    echo ">> [$(date +%H:%M:%S)] START $exp (seed=$seed)"
    echo ">> DF_ARGS=${RUN[$name]}"
    echo "=================================================================="
    EXP_NAME="$exp" DATA_DIR="$DATA" DF_ARGS="${RUN[$name]}" SEED="$seed" \
      bash "$HERE/train_one_7b.sh" > "$LOGDIR/$exp.log" 2>&1
    echo ">> [$(date +%H:%M:%S)] END $exp (exit $?)"
    ray stop --force >/dev/null 2>&1 || true
    pkill -9 -f raylet 2>/dev/null || true; pkill -9 -f "ray::" 2>/dev/null || true
    sleep 10
  done
done
echo ">> [$(date +%H:%M:%S)] ALL MULTI-SEED RUNS DONE"
