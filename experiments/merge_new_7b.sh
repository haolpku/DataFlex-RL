#!/usr/bin/env bash
# Merge FSDP shards -> HF safetensors for the 6 new-algorithm 7B checkpoints,
# so they can be evaluated by eval_7b_parallel.sh. Idempotent: skips any run whose
# huggingface/ already has weights.
set -uo pipefail
ROOT=/apdcephfs_zwfy14/share_304380933/aldenliang
CKPT=$ROOT/df_ckpts_7b
STEP=global_step_300

MC=$ROOT/miniconda3
source "$MC/etc/profile.d/conda.sh"
conda activate verl
cd "$ROOT/frameworks/verl"

RUNS="${*:-gfpo maxvar topk per softmax diffband}"
for run in $RUNS; do
  ACTOR="$CKPT/$run/$STEP/actor"
  HFDIR="$ACTOR/huggingface"
  if ls "$HFDIR"/*.safetensors >/dev/null 2>&1; then
    echo ">> [$(date +%H:%M:%S)] $run already merged, skip"
    continue
  fi
  echo ">> [$(date +%H:%M:%S)] merging $run -> $HFDIR"
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$ACTOR" \
    --target_dir "$HFDIR" 2>&1 | grep -viE "pynvml|FutureWarning|NPU not support" | tail -3
  echo ">> [$(date +%H:%M:%S)] done $run"
done
echo ">> ALL MERGES DONE"
