#!/usr/bin/env bash
# Mixer-round worker: same file-lock queue design as queue_worker.sh but hardcoded to
# the mix trainer + mix ckpt roots. Jobs come from queue/jobs_mix.txt (scale|name|seed|DF_ARGS).
set -uo pipefail
ROOT=/jizhicfs/aldenliang
QROOT=$ROOT/queue
JOBS=$QROOT/jobs_mix.txt
CLAIMS=$QROOT/claims_mix
LOGDIR=$ROOT/df_logs_mix_seeds
mkdir -p "$CLAIMS" "$LOGDIR"
HOST=$(hostname)-$$

driver="$ROOT/DataFlex-RL/experiments/train_one_mix_jizhi.sh"
model_for(){ case "$1" in 7b) echo "$ROOT/models/Qwen2.5-7B-Instruct";; 05b) echo "$ROOT/models/Qwen2.5-0.5B-Instruct";; *) echo "";; esac; }
ckproot_for(){ case "$1" in 7b) echo "$ROOT/campaign_v2/mix_7b";; 05b) echo "$ROOT/campaign_v2/mix_05b";; *) echo "";; esac; }

log(){ echo "[$(date +%H:%M:%S)][$HOST] $*"; }
log "mixer worker up. jobs=$JOBS"

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

  if [ -z "$claimed" ]; then log "no claimable mixer jobs -> exiting"; break; fi

  MODEL="$(model_for "$CSCALE")"
  log "START $claimed  DF_ARGS=[$CDFARGS]"
  source "$ROOT/DataFlex-RL/experiments/df_gpu_cleanup.sh" 2>/dev/null && df_gpu_cleanup || true
  EXP_NAME="${CNAME}_s${CSEED}" DATA_DIR="$ROOT/data/multidomain_3" DF_ARGS="$CDFARGS" SEED="$CSEED" \
    CKPT_ROOT="$CCKROOT" MODEL="$MODEL" \
    bash "$driver" > "$LOGDIR/$claimed.log" 2>&1
  rc=$?
  log "END $claimed rc=$rc"
  if [ "$rc" -eq 0 ] && [ -d "$CCKROOT/${CNAME}_s${CSEED}/global_step_300" ]; then
    echo "done rc=0 $(date -Iseconds)" > "$CLAIMS/$claimed/status"
  else
    echo "failed rc=$rc $(date -Iseconds)" > "$CLAIMS/$claimed/status"
    mv "$CLAIMS/$claimed" "$CLAIMS/${claimed}.failed.$$" 2>/dev/null || true
  fi
  source "$ROOT/DataFlex-RL/experiments/df_gpu_cleanup.sh" 2>/dev/null && df_gpu_cleanup || true
done
