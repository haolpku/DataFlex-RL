#!/usr/bin/env bash
# Central file-lock queue worker for the multi-seed 7B campaign (Paper 2).
# 4 H20 boxes share /jizhicfs; each runs one worker; each job grabs all 8 GPUs of its box.
# Atomic claim via `mkdir` (fails if dir exists -> someone else owns it). Idempotent:
# a job whose ckpt (global_step_300) already exists is skipped and marked done.
#
# Usage (on each box):  bash queue_worker.sh
# Job list:  $QROOT/jobs.txt  — one "name|seed|DF_ARGS" per line (blank DF_ARGS = baseline)
set -uo pipefail
ROOT=/jizhicfs/aldenliang
QROOT=$ROOT/queue
JOBS=$QROOT/jobs.txt
CLAIMS=$QROOT/claims          # mkdir-locks live here
LOGDIR=$ROOT/df_logs_7b_seeds
CKPT_ROOT=$ROOT/df_ckpts_7b_seeds
DRIVER=$ROOT/DataFlex-RL/experiments/train_one_7b_jizhi.sh
mkdir -p "$CLAIMS" "$LOGDIR"
HOST=$(hostname)-$$

log(){ echo "[$(date +%H:%M:%S)][$HOST] $*"; }

log "worker up. jobs=$JOBS"
while :; do
  claimed=""
  # iterate the job list; try to claim the first unclaimed, unfinished job
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    case "$line" in \#*) continue;; esac
    name="${line%%|*}"; rest="${line#*|}"; seed="${rest%%|*}"; dfargs="${rest#*|}"
    exp="${name}_s${seed}"
    # already trained?
    if [ -d "$CKPT_ROOT/$exp/global_step_300" ]; then continue; fi
    # try atomic claim
    if mkdir "$CLAIMS/$exp" 2>/dev/null; then
      echo "$HOST $(date -Iseconds)" > "$CLAIMS/$exp/owner"
      claimed="$exp"; CNAME="$name"; CSEED="$seed"; CDFARGS="$dfargs"
      break
    fi
  done < "$JOBS"

  if [ -z "$claimed" ]; then
    log "no claimable jobs left -> exiting worker"
    break
  fi

  log "START $claimed  DF_ARGS=[$CDFARGS]"
  EXP_NAME="$claimed" DATA_DIR="$ROOT/data/dapo_math" DF_ARGS="$CDFARGS" SEED="$CSEED" \
    CKPT_ROOT="$CKPT_ROOT" \
    bash "$DRIVER" > "$LOGDIR/$claimed.log" 2>&1
  rc=$?
  log "END $claimed rc=$rc"
  if [ "$rc" -eq 0 ] && [ -d "$CKPT_ROOT/$claimed/global_step_300" ]; then
    echo "done rc=0 $(date -Iseconds)" > "$CLAIMS/$claimed/status"
  else
    # release the claim so another worker (or a retry pass) can pick it up
    echo "failed rc=$rc $(date -Iseconds)" > "$CLAIMS/$claimed/status"
    mv "$CLAIMS/$claimed" "$CLAIMS/${claimed}.failed.$$" 2>/dev/null || true
  fi
  # cleanup ray between jobs
  ray stop --force >/dev/null 2>&1 || true
  pkill -9 -f raylet 2>/dev/null || true; pkill -9 -f "ray::" 2>/dev/null || true
  sleep 10
done
