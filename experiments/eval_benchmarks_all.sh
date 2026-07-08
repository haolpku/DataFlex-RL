#!/usr/bin/env bash
# Parallel per-benchmark eval runner for campaign_v1 (main + mixer).
# Amortized cold-start: each ckpt loads once and iterates ALL benchmarks (idempotent).
# 8 GPUs per box, 1 ckpt per GPU. Split ckpts across 6 boxes via SCALE/RUNS env.
#
# Benchmarks (declared inside):
#   Math (7): gsm8k / math_500 / minerva_math / olympiadbench / amc23 / aime24 / aime25
#   Science: gpqa_diamond
#   Logic: kk_hard (people7+8, 200)
#
# Usage:
#   SCALE=7b RUNS="baseline_s1 ar_s1 ..." bash eval_benchmarks_all.sh
#   SCALE=05b RUNS="..." bash ...
#   MIX=1 SCALE=7b RUNS="..." bash ...       # eval mixer ckpts under campaign_v1/mix_7b/
set -uo pipefail
ROOT=/jizhicfs/aldenliang
STEP=global_step_300
PY=$ROOT/miniconda3/envs/verl/bin/python
SCRIPT=$ROOT/DataFlex-RL/experiments/eval_benchmark_batch.py
NGPU=${NGPU:-8}
MAXTOK=${MAXTOK:-4096}
export PYTHONUNBUFFERED=1 VLLM_USE_V1=1 TOKENIZERS_PARALLELISM=false

SCALE="${SCALE:-both}"
MIX="${MIX:-0}"

# ckpt-root prefix
if [ "$MIX" = "1" ]; then
  PREFIX_7B="$ROOT/campaign_v1/mix_7b"
  PREFIX_05B="$ROOT/campaign_v1/mix_05b"
else
  PREFIX_7B="$ROOT/campaign_v1/7b"
  PREFIX_05B="$ROOT/campaign_v1/05b"
fi
[ "$SCALE" = "both" ] && SCALES="7b 05b" || SCALES="$SCALE"

# Benchmark file paths (must be readable from the box; use jizhicfs-mirrored copies or apdcephfs originals)
BENCH_ROOT="$ROOT/benchmarks"
BENCHMARKS=(
  "$BENCH_ROOT/math/gsm8k.parquet"
  "$BENCH_ROOT/math/math_500.parquet"
  "$BENCH_ROOT/math/minerva_math.parquet"
  "$BENCH_ROOT/math/olympiadbench.parquet"
  "$BENCH_ROOT/math/amc23.parquet"
  "$BENCH_ROOT/math/aime24.parquet"
  "$BENCH_ROOT/math/aime25.parquet"
  "$BENCH_ROOT/science/gpqa_diamond.parquet"
  "$BENCH_ROOT/logic/kk_hard.parquet"
)

# Enumerate ckpts to eval
JOBS=()
for sc in $SCALES; do
  [ "$sc" = "7b" ] && PFX="$PREFIX_7B" || PFX="$PREFIX_05B"
  if [ -n "${RUNS:-}" ]; then RUN_LIST="$RUNS"; else RUN_LIST=$(ls "$PFX" 2>/dev/null); fi
  for run in $RUN_LIST; do
    HF="$PFX/$run/$STEP/actor/huggingface"
    OUT_DIR="$HF/benchmark_evals"
    if ! ls "$HF"/*.safetensors >/dev/null 2>&1; then
      echo ">> [$sc/$run] NOT merged, skip"; continue
    fi
    # skip if ALL benchmark JSONs exist
    all_done=1
    for bp in "${BENCHMARKS[@]}"; do
      bn=$(basename "$bp" .parquet)
      [ -f "$OUT_DIR/$bn.json" ] || { all_done=0; break; }
    done
    if [ "$all_done" = 1 ]; then echo ">> [$sc/$run] all benchmarks done, skip"; continue; fi
    JOBS+=("$sc/$run")
  done
done
echo ">> ${#JOBS[@]} ckpts to eval across $NGPU GPUs (each iterates $(echo "${BENCHMARKS[*]}" | wc -w) benchmarks)"
LOGDIR=$ROOT/df_logs_bench_eval; mkdir -p "$LOGDIR"

declare -A GPU_PID
for job in "${JOBS[@]}"; do
  sc="${job%%/*}"; run="${job#*/}"
  [ "$sc" = "7b" ] && PFX="$PREFIX_7B" || PFX="$PREFIX_05B"
  HF="$PFX/$run/$STEP/actor/huggingface"
  OUT_DIR="$HF/benchmark_evals"
  # wait for free GPU
  while :; do
    for g in $(seq 0 $((NGPU-1))); do
      pid="${GPU_PID[$g]:-}"
      if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        LOG="$LOGDIR/${sc}_${run}.log"
        echo ">> [$(date +%H:%M:%S)] GPU $g <- $sc/$run"
        VLLM_CACHE_ROOT="/tmp/vllm_bench_${sc}_${run}" \
          $PY "$SCRIPT" --model "$HF" --out_dir "$OUT_DIR" \
              --benchmarks "${BENCHMARKS[@]}" \
              --gpu "$g" --max_tokens "$MAXTOK" \
          > "$LOG" 2>&1 &
        GPU_PID[$g]=$!
        break 2
      fi
    done
    sleep 5
  done
done
for g in $(seq 0 $((NGPU-1))); do
  pid="${GPU_PID[$g]:-}"; [ -n "$pid" ] && wait "$pid" 2>/dev/null || true
done
echo ">> ALL BENCH EVAL JOBS DONE"
