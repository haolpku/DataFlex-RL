#!/usr/bin/env bash
# Merge FSDP shards -> HF safetensors for ALL campaign_v1 ckpts on jizhicfs.
# Idempotent: skips any run whose actor/huggingface/*.safetensors already exists.
# Serial per box (merge is CPU/IO-bound, ~2-5min per ckpt); split across 6 boxes via env.
#
# Usage (on any jizhicfs box):
#   bash merge_campaign_v1.sh                        # merge all 51 ckpts (7b + 05b)
#   SCALE=7b bash merge_campaign_v1.sh               # only 7B
#   SCALE=05b bash merge_campaign_v1.sh              # only 0.5B
#   RUNS="baseline_s1 ar_s2" SCALE=7b bash ...       # specific runs
set -uo pipefail
ROOT=/jizhicfs/aldenliang
STEP=global_step_300
PY=$ROOT/miniconda3/envs/verl/bin/python
export PYTHONUNBUFFERED=1

SCALE="${SCALE:-both}"
[ "$SCALE" = "both" ] && SCALES="7b 05b" || SCALES="$SCALE"

cd "$ROOT/frameworks/verl"
for sc in $SCALES; do
  CKPT="$ROOT/campaign_v1/$sc"
  if [ -n "${RUNS:-}" ]; then RUN_LIST="$RUNS"; else RUN_LIST=$(ls "$CKPT" 2>/dev/null); fi
  for run in $RUN_LIST; do
    ACTOR="$CKPT/$run/$STEP/actor"
    HFDIR="$ACTOR/huggingface"
    if ls "$HFDIR"/*.safetensors >/dev/null 2>&1; then
      echo ">> [$(date +%H:%M:%S)] $sc/$run already merged, skip"
      continue
    fi
    if [ ! -d "$ACTOR" ]; then
      echo ">> [$(date +%H:%M:%S)] $sc/$run: MISSING actor dir, skip"; continue
    fi
    echo ">> [$(date +%H:%M:%S)] merging $sc/$run"
    $PY -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "$ACTOR" \
      --target_dir "$HFDIR" 2>&1 | grep -viE "pynvml|FutureWarning|NPU not support|router_replay" | tail -3
    echo ">> [$(date +%H:%M:%S)] done $sc/$run"
  done
done
echo ">> ALL MERGES DONE"
