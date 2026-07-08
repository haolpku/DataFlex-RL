#!/usr/bin/env bash
# Parallel 3-domain eval for campaign_v1 ckpts on jizhicfs.
# Each ckpt runs on 1 GPU; pool of 8 GPUs per box. Idempotent (skips if eval JSON exists).
# Usage (on any jizhicfs box):
#   bash eval_campaign_v1.sh                            # eval all 51 ckpts (all with safetensors)
#   SCALE=7b bash eval_campaign_v1.sh                   # only 7B
#   SCALE=05b RUNS="baseline_s1 ar_s1" bash eval_...    # subset
set -uo pipefail
ROOT=/jizhicfs/aldenliang
STEP=global_step_300
PY=$ROOT/miniconda3/envs/verl/bin/python
DATA=$ROOT/data/multidomain_3/test.parquet
SCRIPT=$ROOT/DataFlex-RL/experiments/eval_multidomain.py
NGPU=${NGPU:-8}
MAXTOK=${MAXTOK:-4096}
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1 TOKENIZERS_PARALLELISM=false

SCALE="${SCALE:-both}"
[ "$SCALE" = "both" ] && SCALES="7b 05b" || SCALES="$SCALE"

# collect (scale, run) jobs (only merged ckpts)
JOBS=()
for sc in $SCALES; do
  CKPT="$ROOT/campaign_v1/$sc"
  if [ -n "${RUNS:-}" ]; then RUN_LIST="$RUNS"; else RUN_LIST=$(ls "$CKPT" 2>/dev/null); fi
  for run in $RUN_LIST; do
    HF="$CKPT/$run/$STEP/actor/huggingface"
    OUT="$HF/multidomain_eval.json"
    if [ -f "$OUT" ]; then echo ">> [$(date +%H:%M:%S)] $sc/$run already evaluated, skip"; continue; fi
    if ! ls "$HF"/*.safetensors >/dev/null 2>&1; then
      echo ">> [$(date +%H:%M:%S)] $sc/$run: NOT merged yet, skip"; continue
    fi
    JOBS+=("$sc/$run")
  done
done
echo ">> ${#JOBS[@]} eval jobs across $NGPU GPUs"
LOGDIR=$ROOT/df_logs_eval; mkdir -p "$LOGDIR"

# 7B occupies a full GPU (~15G at bf16 + KV cache); 0.5B one per GPU too. 1 job / GPU.
declare -A GPU_PID
for job in "${JOBS[@]}"; do
  sc="${job%%/*}"; run="${job#*/}"
  # wait for a free GPU
  while :; do
    for g in $(seq 0 $((NGPU-1))); do
      pid="${GPU_PID[$g]:-}"
      if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        HF="$ROOT/campaign_v1/$sc/$run/$STEP/actor/huggingface"
        OUT="$HF/multidomain_eval.json"
        LOG="$LOGDIR/${sc}_${run}.log"
        echo ">> [$(date +%H:%M:%S)] GPU $g <- $sc/$run"
        VLLM_CACHE_ROOT="/tmp/vllm_eval_${sc}_${run}" \
          $PY "$SCRIPT" --model "$HF" --data "$DATA" --out "$OUT" \
              --gpu "$g" --max_tokens "$MAXTOK" \
          > "$LOG" 2>&1 &
        GPU_PID[$g]=$!
        break 2   # break out of inner for + `while`
      fi
    done
    sleep 5
  done
done
# wait for last batch to drain
for g in $(seq 0 $((NGPU-1))); do
  pid="${GPU_PID[$g]:-}"; [ -n "$pid" ] && wait "$pid" 2>/dev/null || true
done
echo ">> ALL EVAL JOBS DONE"
