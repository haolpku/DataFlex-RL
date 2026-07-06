#!/usr/bin/env bash
# Parallel 7B eval: 3 runs x 4 datasets across 8 GPUs.
# Each (run,dataset) is one math_eval.py call pinned to one GPU. A simple GPU pool
# keeps all 8 cards busy until the 12 jobs drain.
#
# eval max_tokens=4096 (competition-math safe; see STEER/docs/rollout_length.md).
set -uo pipefail

ROOT=/apdcephfs_zwfy14/share_304380933/aldenliang
CKPT=$ROOT/df_ckpts_7b
STEP=global_step_300
QWEN_EVAL=$ROOT/frameworks/Qwen2.5-Math/evaluation
LOGDIR=$ROOT/df_logs_7b/eval
mkdir -p "$LOGDIR"

RUNS="${RUNS:-baseline ar difffilter}"
DATASETS="${DATASETS:-gsm8k math amc23 aime24}"
MAXTOK=4096
NGPU=8

MC=$ROOT/miniconda3
source "$MC/etc/profile.d/conda.sh"
conda activate qwen-eval
export VLLM_USE_V1=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export http_proxy="http://hy-proxy.woa.com:3128" https_proxy="http://hy-proxy.woa.com:3128"
export no_proxy=".woa.com,mirrors.cloud.tencent.com,localhost,127.0.0.1"

# build the job list: "run dataset"
JOBS=()
for r in $RUNS; do for d in $DATASETS; do JOBS+=("$r $d"); done; done
echo ">> ${#JOBS[@]} jobs across $NGPU GPUs, max_tokens=$MAXTOK"

run_one() {
  local gpu="$1" run="$2" ds="$3"
  local HF="$CKPT/$run/$STEP/actor/huggingface"
  # Per-job vLLM compile cache: parallel jobs sharing /root/.cache/vllm collide on
  # torch_compile temp files (FileNotFoundError). Isolate each job's cache root.
  local VC="/tmp/vllm_cache_${run}_${ds}"
  VLLM_CACHE_ROOT="$VC" CUDA_VISIBLE_DEVICES=$gpu taskset -c $((gpu*8))-$((gpu*8+7)) \
    python -u "$QWEN_EVAL/math_eval.py" \
      --model_name_or_path "$HF" \
      --data_names "$ds" \
      --data_dir "$QWEN_EVAL/data" \
      --output_dir "$HF/math_eval" \
      --split test --prompt_type qwen25-math-cot \
      --num_test_sample -1 \
      --max_tokens_per_call $MAXTOK \
      --seed 0 --temperature 0 --top_p 1 \
      --use_vllm --save_outputs --overwrite --apply_chat_template \
      > "$LOGDIR/${run}_${ds}.log" 2>&1
}

# GPU pool: launch up to NGPU at once, refill as they finish
declare -A GPU_PID   # gpu -> pid
next=0
for job in "${JOBS[@]}"; do
  set -- $job; jr="$1"; jd="$2"
  # find a free gpu (wait if all busy)
  while :; do
    for g in $(seq 0 $((NGPU-1))); do
      pid=${GPU_PID[$g]:-}
      if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        run_one "$g" "$jr" "$jd" &
        GPU_PID[$g]=$!
        echo ">> [$(date +%H:%M:%S)] GPU$g <- $jr/$jd (pid ${GPU_PID[$g]})"
        job=""; break
      fi
    done
    [ -z "$job" ] && break
    sleep 5
  done
done
# wait for all
wait
echo ">> [$(date +%H:%M:%S)] ALL EVAL DONE"

# collect
echo "======== 7B RESULTS (step 300, max_tokens=$MAXTOK) ========"
printf "%-12s %-10s %-10s %-10s %-10s\n" run gsm8k math amc23 aime24
for r in $RUNS; do
  line="$r"
  for d in $DATASETS; do
    m=$(ls "$CKPT/$r/$STEP/actor/huggingface/math_eval/$d/"*_metrics.json 2>/dev/null | head -1)
    acc=$(python -c "import json;print(json.load(open('$m'))['acc'])" 2>/dev/null || echo "-")
    line="$line $acc"
  done
  echo "$line"
done | column -t
