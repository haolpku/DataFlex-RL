#!/usr/bin/env bash
# Orchestrate the 4-way DataFlex comparison, SERIALLY, each on all 8 GPUs.
#
#   baseline  : stock verl GRPO (no DataFlex)
#   reweight  : advantage_magnitude -> softmax per-sample loss weights
#   select    : group_solve_rate -> threshold_band (DAPO-style filtering)
#   mix       : reward_difficulty -> reward_gap domain proportions (2-domain data)
#
# All share model/data/optimizer/batch/seed; only the DataFlex block differs.
# Checkpoints -> $ROOT/df_ckpts/<exp>, logs -> $ROOT/df_logs/<exp>.log
#
# Usage:  bash experiments/run_all.sh            # all four, in order
#         bash experiments/run_all.sh reweight   # just one (or a subset, space-sep)
set -uo pipefail

ROOT=/apdcephfs_zwfy14/share_304380933/aldenliang
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR=$ROOT/df_logs
mkdir -p "$LOGDIR"

export TOTAL_STEPS=${TOTAL_STEPS:-300}
export SAVE_FREQ=${SAVE_FREQ:-100}
export CKPT_ROOT=$ROOT/df_ckpts

run() {
  local name="$1"; shift
  local data="$1"; shift
  local df="$1"; shift
  echo "=================================================================="
  echo ">> [$(date +%H:%M:%S)] START run: $name"
  echo ">> data=$data"
  echo ">> DF_ARGS=$df"
  echo ">> log=$LOGDIR/$name.log"
  echo "=================================================================="
  EXP_NAME="$name" DATA_DIR="$data" DF_ARGS="$df" \
    bash "$HERE/train_one.sh" > "$LOGDIR/$name.log" 2>&1
  local rc=$?
  echo ">> [$(date +%H:%M:%S)] END run: $name (exit $rc)"
  # clean up ray between runs so the next one starts fresh
  ray stop --force >/dev/null 2>&1 || true
  pkill -9 -f raylet 2>/dev/null || true
  pkill -9 -f "ray::" 2>/dev/null || true
  sleep 8
  return $rc
}

# --- run definitions ---
DATA_STD=$ROOT/data/gsm8k
DATA_MIX=$ROOT/data/gsm8k_2domain

REWEIGHT_DF="trainer.v1.trainer_mode=dataflex_sync +dataflex.mechanism=reweight +dataflex.scorer.name=advantage_magnitude +dataflex.scorer.params.agg=mean +dataflex.actuator.name=softmax +dataflex.actuator.params.temperature=1.0 +dataflex.warmup_step=0"

SELECT_DF="trainer.v1.trainer_mode=dataflex_sync +dataflex.mechanism=select +dataflex.scorer.name=group_solve_rate +dataflex.scorer.params.success_threshold=0.5 +dataflex.actuator.name=threshold_band +dataflex.actuator.params.low=0.0 +dataflex.actuator.params.high=1.0 +dataflex.warmup_step=0"

MIX_DF="trainer.v1.trainer_mode=dataflex_mix_sync trainer.v1.sampler.custom_sampler.path=pkg://dataflex_verl.replay_buffer trainer.v1.sampler.custom_sampler.name=DataFlexMixReplayBuffer +dataflex.mechanism=mix +dataflex.domains=[gsm8k_short,gsm8k_long] +dataflex.scorer.name=reward_difficulty +dataflex.actuator.name=reward_gap +dataflex.actuator.params.temperature=1.0 +dataflex.actuator.params.floor=0.05 +dataflex.warmup_step=10 +dataflex.update_step=5 +dataflex.window=50"

WANT="${*:-baseline reweight select mix}"
for w in $WANT; do
  case "$w" in
    baseline) run baseline "$DATA_STD" "" ;;
    reweight) run reweight "$DATA_STD" "$REWEIGHT_DF" ;;
    select)   run select   "$DATA_STD" "$SELECT_DF" ;;
    mix)      run mix       "$DATA_MIX" "$MIX_DF" ;;
    *) echo "unknown run: $w"; exit 2 ;;
  esac
done
echo ">> [$(date +%H:%M:%S)] ALL DONE. checkpoints in $CKPT_ROOT, logs in $LOGDIR"
