#!/usr/bin/env bash
# Watchdog to run overnight while user sleeps:
#   1. Poll v2 eval progress. Reruns aggregate every 10 min.
#   2. When eval math ≥50/52 AND gpqa ≥50/52, launch mixer round.
#   3. Log everything under /jizhicfs/aldenliang/df_watchdog.log
set -uo pipefail
ROOT=/jizhicfs/aldenliang
LOG=$ROOT/df_watchdog.log
STATE=$ROOT/df_watchdog.state
BOX=$(hostname -I | awk '{print $1}')

log(){ echo "[$(date '+%m-%d %H:%M:%S')][$BOX] $*" | tee -a "$LOG"; }

log "watchdog up. mode=eval-wait"
echo "eval-wait" > "$STATE"

while true; do
  # Aggregate eval progress
  math_done=$(find $ROOT/frameworks/Qwen2.5-Math/evaluation/outputs/jizhicfs/aldenliang/campaign_v2 \
    -name '*_metrics.json' 2>/dev/null | \
    awk -F/ '{print $(NF-7)"__"$(NF-6)}' | sort -u | wc -l)
  gpqa_done=$(ls $ROOT/queue_eval_gpqa/results/*.csv 2>/dev/null | wc -l)
  log "eval progress: math_ckpts=$math_done gpqa_ckpts=$gpqa_done (target=54 each)"

  mode=$(cat "$STATE" 2>/dev/null || echo "eval-wait")

  if [ "$mode" = "eval-wait" ]; then
    # If both hit ≥50, launch mixer
    if [ "$math_done" -ge 50 ] && [ "$gpqa_done" -ge 50 ]; then
      log "eval essentially done, LAUNCHING MIXER"
      # Generate jobs
      bash $ROOT/DataFlex-RL/experiments/gen_jobs_mix.sh >> "$LOG" 2>&1
      log "wrote mixer jobs list"
      # Kill any remaining math/gpqa workers to free GPUs
      for H in 29.164.4.21 29.164.3.167 28.49.148.180 28.49.144.84 28.49.144.87 29.164.0.198; do
        (expect /tmp/df_ssh.exp $H "pkill -9 -f batch_math_worker; pkill -9 -f batch_gpqa_worker; pkill -9 -f eval_math.sh; pkill -9 -f eval_oc.sh; pkill -9 -f 'python.*eval_math'; pkill -9 -f 'python.*run.py.*opencompass'; nvidia-smi | head -20" 2>&1 | tail -5) &
      done
      wait
      log "eval workers killed on all 6 boxes"
      # Launch mixer workers on all 6 boxes
      for H in 29.164.4.21 29.164.3.167 28.49.148.180 28.49.144.84 28.49.144.87 29.164.0.198; do
        expect /tmp/df_ssh.exp $H "cd $ROOT/DataFlex-RL/experiments && setsid nohup bash queue_worker_mix.sh > $ROOT/df_mix_worker.log 2>&1 </dev/null & echo LAUNCHED_$H" 2>&1 | grep -oE "LAUNCHED_[0-9.]+" | head -1 | tee -a "$LOG"
      done
      log "mixer round launched on 6 boxes"
      echo "mixer-running" > "$STATE"
    fi
  elif [ "$mode" = "mixer-running" ]; then
    mix_done=$(ls -d $ROOT/campaign_v2/mix_7b/*_s?/global_step_300 $ROOT/campaign_v2/mix_05b/*_s?/global_step_300 2>/dev/null | wc -l)
    log "mixer progress: mix_ckpts=$mix_done (target=24)"
    if [ "$mix_done" -ge 22 ]; then
      log "mixer essentially done. exiting watchdog"
      echo "done" > "$STATE"
      break
    fi
  fi

  sleep 600  # 10 min
done
log "watchdog exit"
