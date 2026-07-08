#!/usr/bin/env bash
# Generate the MIXER-round job list (separate from the main select/reweight campaign,
# because mix uses the dataflex_mix_sync trainer + custom sampler). 3 mixers x 2 scales
# x 3 seeds = 18 jobs. Format: scale|name|seed|DF_ARGS  (scale = 7b|05b).
# Runs on the 3-domain multidomain_3 set — where dynamic mixture actually has room to work.
set -euo pipefail
QROOT=/jizhicfs/aldenliang/queue
mkdir -p "$QROOT"
OUT=$QROOT/jobs_mix.txt

# all mixers steer from per-domain sliding-window reward (reward_difficulty scorer)
COMMON="+dataflex.scorer.name=reward_difficulty +dataflex.warmup_step=1 +dataflex.update_step=1 +dataflex.window=50"
declare -A RUN
RUN[reward_gap]="$COMMON +dataflex.actuator.name=reward_gap +dataflex.actuator.params.temperature=1.0 +dataflex.actuator.params.floor=0.05"
RUN[dump_ucb]="$COMMON +dataflex.actuator.name=dump_ucb +dataflex.actuator.params.temperature=1.0 +dataflex.actuator.params.c=1.0 +dataflex.actuator.params.floor=0.05"
RUN[tscl]="$COMMON +dataflex.actuator.name=tscl +dataflex.actuator.params.temperature=1.0 +dataflex.actuator.params.floor=0.05"
# static uniform = the fixed-ratio control (mixture's honest baseline)
RUN[static]="+dataflex.scorer.name=reward_difficulty +dataflex.actuator.name=static +dataflex.warmup_step=1 +dataflex.update_step=1 +dataflex.window=50"

MIXERS="${MIXERS:-reward_gap dump_ucb tscl static}"
SEEDS="${SEEDS:-1 2 3}"
SCALES="${SCALES:-7b 05b}"

: > "$OUT"
for scale in $SCALES; do
  for seed in $SEEDS; do
    for name in $MIXERS; do
      printf '%s|%s|%s|%s\n' "$scale" "$name" "$seed" "${RUN[$name]}" >> "$OUT"
    done
  done
done
echo "wrote $(wc -l < "$OUT") mixer jobs to $OUT"
