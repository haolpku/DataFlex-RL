#!/usr/bin/env bash
# Evaluate the 4 DataFlex comparison runs (baseline/reweight/select/mix) at step 300
# on the math benchmark suite. Two phases per run:
#   1. merge verl FSDP shards -> HF safetensors (into the checkpoint's huggingface/ dir)
#   2. run the Qwen2.5-Math harness on gsm8k (in-distribution) + math_500 + amc23 (generalization)
#
# Serial, single GPU (eval is light). Results land under
#   $CKPT/<run>/global_step_300/actor/huggingface/math_eval/<dataset>/*_metrics.json
set -uo pipefail

ROOT=/apdcephfs_zwfy14/share_304380933/aldenliang
CKPT=$ROOT/df_ckpts
STEP=${STEP:-global_step_300}
DATASETS=${DATASETS:-"gsm8k,math,amc23"}
EVAL_GPU=${EVAL_GPU:-0}

MC=$ROOT/miniconda3
source "$MC/etc/profile.d/conda.sh"

RUNS="${*:-baseline reweight select mix}"

for run in $RUNS; do
  ACTOR="$CKPT/$run/$STEP/actor"
  HFDIR="$ACTOR/huggingface"
  echo "=================================================================="
  echo ">> [$(date +%H:%M:%S)] EVAL run=$run  step=$STEP"
  echo "=================================================================="

  # ---- phase 1: merge FSDP -> HF (skip if weights already present) ----
  if ls "$HFDIR"/*.safetensors >/dev/null 2>&1 || ls "$HFDIR"/model*.bin >/dev/null 2>&1; then
    echo ">> HF weights already present, skip merge"
  else
    echo ">> merging FSDP shards -> $HFDIR"
    conda activate verl
    cd "$ROOT/frameworks/verl"
    python -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "$ACTOR" \
      --target_dir "$HFDIR" 2>&1 | grep -viE "pynvml|FutureWarning|RouterReplay|NPU not support" | tail -5
    conda deactivate
  fi

  # ---- phase 2: math eval on the merged HF model ----
  echo ">> running math eval on $DATASETS"
  CUDA_VISIBLE_DEVICES=$EVAL_GPU bash "$ROOT/benchmarks/scripts/eval_math.sh" "$HFDIR" "$DATASETS" -1 \
    > "$ROOT/df_logs/eval_${run}.log" 2>&1
  echo ">> [$(date +%H:%M:%S)] done run=$run  (log: df_logs/eval_${run}.log)"
done

echo ""
echo ">> ALL EVAL DONE. Collecting scores:"
for run in $RUNS; do
  echo "--- $run ---"
  for d in ${DATASETS//,/ }; do
    m=$(ls "$CKPT/$run/$STEP/actor/huggingface/math_eval/$d/"*_metrics.json 2>/dev/null | head -1)
    if [ -n "$m" ]; then
      acc=$(python -c "import json,sys; print(json.load(open('$m')).get('acc','?'))" 2>/dev/null)
      echo "  $d: acc=$acc"
    else
      echo "  $d: (no metrics)"
    fi
  done
done
