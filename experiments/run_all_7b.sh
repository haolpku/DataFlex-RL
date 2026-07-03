#!/usr/bin/env bash
# Orchestrate the 7B DataFlex comparison on dapo-math-17k, SERIALLY (8 GPU each).
# Runs the 3 configs that are meaningful on single-domain math data:
#   baseline  : stock verl GRPO
#   ar        : Advantage Reweighting (token_prob -> advantage_reweight)
#   difffilter: Online Difficulty Filtering (group_solve_rate -> threshold_band 0.2-0.8)
# (mix/DUMP is NOT run here — single domain has nothing to mix; needs >=3-domain data.)
#
# Usage: bash experiments/run_all_7b.sh            # all three
#        bash experiments/run_all_7b.sh ar         # subset
set -uo pipefail

ROOT=/apdcephfs_zwfy14/share_304380933/aldenliang
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR=$ROOT/df_logs_7b
mkdir -p "$LOGDIR"
export TOTAL_STEPS=${TOTAL_STEPS:-300}
export SAVE_FREQ=${SAVE_FREQ:-100}
export CKPT_ROOT=$ROOT/df_ckpts_7b
DATA=$ROOT/data/dapo_math

run() {
  local name="$1"; shift
  local df="$1"; shift
  echo "=================================================================="
  echo ">> [$(date +%H:%M:%S)] START 7B run: $name"
  echo ">> DF_ARGS=$df  log=$LOGDIR/$name.log"
  echo "=================================================================="
  EXP_NAME="$name" DATA_DIR="$DATA" DF_ARGS="$df" \
    bash "$HERE/train_one_7b.sh" > "$LOGDIR/$name.log" 2>&1
  echo ">> [$(date +%H:%M:%S)] END 7B run: $name (exit $?)"
  ray stop --force >/dev/null 2>&1 || true
  pkill -9 -f raylet 2>/dev/null || true; pkill -9 -f "ray::" 2>/dev/null || true
  sleep 10
}

AR_DF="trainer.v1.trainer_mode=dataflex_sync +dataflex.mechanism=reweight +dataflex.scorer.name=token_prob +dataflex.actuator.name=advantage_reweight +dataflex.actuator.params.alpha=0.5 +dataflex.warmup_step=0"

DIFF_DF="trainer.v1.trainer_mode=dataflex_sync +dataflex.mechanism=select +dataflex.scorer.name=group_solve_rate +dataflex.scorer.params.success_threshold=0.5 +dataflex.actuator.name=threshold_band +dataflex.actuator.params.low=0.2 +dataflex.actuator.params.high=0.8 +dataflex.warmup_step=0"

WANT="${*:-baseline ar difffilter}"
for w in $WANT; do
  case "$w" in
    baseline)   run baseline "" ;;
    ar)         run ar "$AR_DF" ;;
    difffilter) run difffilter "$DIFF_DF" ;;
    *) echo "unknown run: $w"; exit 2 ;;
  esac
done
echo ">> [$(date +%H:%M:%S)] 7B ALL DONE. ckpts=$CKPT_ROOT logs=$LOGDIR"
