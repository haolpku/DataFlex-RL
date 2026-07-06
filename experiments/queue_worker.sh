#!/usr/bin/env bash
# Central file-lock queue worker for the multi-seed campaign (Paper 2).
# H20 boxes share /jizhicfs; each runs one worker; each job grabs all 8 GPUs of its box.
# Atomic claim via `mkdir` (fails if dir exists -> someone else owns it). Idempotent:
# a job whose ckpt (global_step_300) already exists is skipped.
#
# Usage (on each box):  bash queue_worker.sh
# Job list:  $QROOT/jobs.txt  — one "scale|name|seed|DF_ARGS" per line (scale = 7b|05b)
set -uo pipefail
ROOT=/jizhicfs/aldenliang
QROOT=$ROOT/queue
JOBS=$QROOT/jobs.txt
CLAIMS=$QROOT/claims          # mkdir-locks live here
LOGDIR=$ROOT/df_logs_seeds
mkdir -p "$CLAIMS" "$LOGDIR"
HOST=$(hostname)-$$

# per-scale driver + ckpt root
driver_for(){ case "$1" in 7b) echo "$ROOT/DataFlex-RL/experiments/train_one_7b_jizhi.sh";; 05b) echo "$ROOT/DataFlex-RL/experiments/train_one_05b_jizhi.sh";; *) echo "";; esac; }
ckproot_for(){ case "$1" in 7b) echo "$ROOT/df_ckpts_7b_seeds";; 05b) echo "$ROOT/df_ckpts_05b_seeds";; *) echo "";; esac; }

log(){ echo "[$(date +%H:%M:%S)][$HOST] $*"; }
log "worker up. jobs=$JOBS"

while :; do
  claimed=""
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    case "$line" in \#*) continue;; esac
    scale="${line%%|*}"; r1="${line#*|}"; name="${r1%%|*}"; r2="${r1#*|}"; seed="${r2%%|*}"; dfargs="${r2#*|}"
    exp="${scale}_${name}_s${seed}"
    ckroot="$(ckproot_for "$scale")"
    [ -z "$ckroot" ] && { log "bad scale in: $line"; continue; }
    if [ -d "$ckroot/${name}_s${seed}/global_step_300" ]; then continue; fi
    if mkdir "$CLAIMS/$exp" 2>/dev/null; then
      echo "$HOST $(date -Iseconds)" > "$CLAIMS/$exp/owner"
      claimed="$exp"; CSCALE="$scale"; CNAME="$name"; CSEED="$seed"; CDFARGS="$dfargs"; CCKROOT="$ckroot"
      break
    fi
  done < "$JOBS"

  if [ -z "$claimed" ]; then log "no claimable jobs left -> exiting"; break; fi

  drv="$(driver_for "$CSCALE")"
  log "START $claimed  driver=$(basename "$drv")  DF_ARGS=[$CDFARGS]"
  EXP_NAME="${CNAME}_s${CSEED}" DATA_DIR="$ROOT/data/multidomain_3" DF_ARGS="$CDFARGS" SEED="$CSEED" \
    CKPT_ROOT="$CCKROOT" \
    bash "$drv" > "$LOGDIR/$claimed.log" 2>&1
  rc=$?
  log "END $claimed rc=$rc"
  if [ "$rc" -eq 0 ] && [ -d "$CCKROOT/${CNAME}_s${CSEED}/global_step_300" ]; then
    echo "done rc=0 $(date -Iseconds)" > "$CLAIMS/$claimed/status"
  else
    echo "failed rc=$rc $(date -Iseconds)" > "$CLAIMS/$claimed/status"
    mv "$CLAIMS/$claimed" "$CLAIMS/${claimed}.failed.$$" 2>/dev/null || true
  fi
  ray stop --force >/dev/null 2>&1 || true
  pkill -9 -f raylet 2>/dev/null || true; pkill -9 -f "ray::" 2>/dev/null || true
  sleep 10
done
