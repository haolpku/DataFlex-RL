#!/usr/bin/env bash
# Generate the multi-seed job list for the queue (Paper 2 statistical backbone).
# 9 algos x seeds{2,3} = 18 jobs. Format per line:  name|seed|DF_ARGS
# DF_ARGS mirror experiments/run_seeds_7b.sh exactly (same unified setting as seed-1).
set -euo pipefail
QROOT=/jizhicfs/aldenliang/queue
mkdir -p "$QROOT"
OUT=$QROOT/jobs.txt
DF="trainer.v1.trainer_mode=dataflex_sync"

declare -A RUN
RUN[baseline]=""
RUN[ar]="$DF +dataflex.mechanism=reweight +dataflex.scorer.name=token_prob +dataflex.actuator.name=advantage_reweight +dataflex.actuator.params.alpha=0.5 +dataflex.warmup_step=0"
RUN[difffilter]="$DF +dataflex.mechanism=select +dataflex.scorer.name=group_solve_rate +dataflex.scorer.params.success_threshold=0.5 +dataflex.actuator.name=threshold_band +dataflex.actuator.params.low=0.2 +dataflex.actuator.params.high=0.8 +dataflex.warmup_step=0"
RUN[gfpo]="$DF +dataflex.mechanism=select +dataflex.scorer.name=reward_difficulty +dataflex.actuator.name=gfpo +dataflex.actuator.params.k=3 +dataflex.actuator.params.metric=efficiency +dataflex.warmup_step=0"
RUN[maxvar]="$DF +dataflex.mechanism=select +dataflex.scorer.name=reward_difficulty +dataflex.actuator.name=max_variance +dataflex.actuator.params.keep_fraction=0.5 +dataflex.warmup_step=0"
RUN[topk]="$DF +dataflex.mechanism=select +dataflex.scorer.name=advantage_magnitude +dataflex.actuator.name=topk_fraction +dataflex.actuator.params.fraction=0.5 +dataflex.warmup_step=0"
RUN[per]="$DF +dataflex.mechanism=reweight +dataflex.scorer.name=advantage_magnitude +dataflex.actuator.name=per_advantage +dataflex.actuator.params.alpha=0.5 +dataflex.warmup_step=0"
RUN[softmax]="$DF +dataflex.mechanism=reweight +dataflex.scorer.name=advantage_magnitude +dataflex.actuator.name=softmax +dataflex.actuator.params.temperature=1.0 +dataflex.warmup_step=0"
RUN[diffband]="$DF +dataflex.mechanism=reweight +dataflex.scorer.name=reward_difficulty +dataflex.actuator.name=difficulty_band +dataflex.warmup_step=0"

ALGOS="baseline ar difffilter gfpo maxvar topk per softmax diffband"
SEEDS="${SEEDS:-2 3}"

: > "$OUT"
for seed in $SEEDS; do
  for name in $ALGOS; do
    printf '%s|%s|%s\n' "$name" "$seed" "${RUN[$name]}" >> "$OUT"
  done
done
echo "wrote $(wc -l < "$OUT") jobs to $OUT"
