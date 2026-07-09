#!/usr/bin/env bash
# Local-box (apdcephfs) share of the campaign: 6 runs = 0.5B seed-3 x {baseline,ar,
# difffilter,gfpo,maxvar,topk}. Serial, 8 GPUs each. Idempotent (skips done ckpts).
# These 6 are RESERVED in the jizhicfs queue so the 6-box cluster won't duplicate them.
set -uo pipefail
ROOT=/apdcephfs_zwfy14/share_304380933/aldenliang
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR=$ROOT/df_logs_local; mkdir -p "$LOGDIR"
CKPT_ROOT=$ROOT/campaign_v2_local/05b
DF="trainer.v1.trainer_mode=dataflex_sync"
declare -A RUN
RUN[baseline]=""
RUN[ar]="$DF +dataflex.mechanism=reweight +dataflex.scorer.name=token_prob +dataflex.actuator.name=advantage_reweight +dataflex.actuator.params.alpha=0.5 +dataflex.warmup_step=0"
RUN[difffilter]="$DF +dataflex.mechanism=select +dataflex.scorer.name=group_solve_rate +dataflex.scorer.params.success_threshold=0.5 +dataflex.actuator.name=threshold_band +dataflex.actuator.params.low=0.2 +dataflex.actuator.params.high=0.8 +dataflex.warmup_step=0"
RUN[gfpo]="$DF +dataflex.mechanism=select +dataflex.scorer.name=reward_difficulty +dataflex.actuator.name=gfpo +dataflex.actuator.params.k=3 +dataflex.actuator.params.metric=efficiency +dataflex.warmup_step=0"
RUN[maxvar]="$DF +dataflex.mechanism=select +dataflex.scorer.name=reward_difficulty +dataflex.actuator.name=max_variance +dataflex.actuator.params.keep_fraction=0.5 +dataflex.warmup_step=0"
RUN[topk]="$DF +dataflex.mechanism=select +dataflex.scorer.name=advantage_magnitude +dataflex.actuator.name=topk_fraction +dataflex.actuator.params.fraction=0.5 +dataflex.warmup_step=0"
SEED=3
for name in baseline ar difffilter gfpo maxvar topk; do
  exp="${name}_s${SEED}"
  if [ -d "$CKPT_ROOT/$exp/global_step_300" ]; then echo ">> $exp DONE, skip"; continue; fi
  echo "==== [$(date +%H:%M:%S)] START $exp ===="
  source "$HERE/df_gpu_cleanup.sh" 2>/dev/null && df_gpu_cleanup || true
  EXP_NAME="$exp" DATA_DIR="$ROOT/data/multidomain_3" DF_ARGS="${RUN[$name]}" SEED="$SEED" \
    CKPT_ROOT="$CKPT_ROOT" bash "$HERE/train_one_05b_local.sh" > "$LOGDIR/$exp.log" 2>&1
  echo "==== [$(date +%H:%M:%S)] END $exp rc=$? ===="
  source "$HERE/df_gpu_cleanup.sh" 2>/dev/null && df_gpu_cleanup || { ray stop --force >/dev/null 2>&1 || true; pkill -9 -f raylet 2>/dev/null || true; sleep 8; }
done
echo ">> LOCAL 6 DONE"
