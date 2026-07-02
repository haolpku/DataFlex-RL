#!/usr/bin/env bash
# One-command DataFlex-verl demo. Runs a mechanism end-to-end on GRPO and prints
# the DataFlex metric so you see the effect without scrolling the full log.
#
# Usage:
#   bash examples/run_demo.sh reweight   # per-sample loss weighting
#   bash examples/run_demo.sh select     # DAPO-style group filtering
#   bash examples/run_demo.sh mix        # dynamic domain proportions
#
# Env overrides: CUDA_VISIBLE_DEVICES (default 0-7). For `mix`, build the 2-domain
# data first (see examples/build_2domain_gsm8k.py / docs/QUICKSTART.md).
set -euo pipefail

MECH="${1:-reweight}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="/tmp/dataflex_demo_${MECH}.log"

case "$MECH" in
  reweight) SCRIPT="run_reweight_grpo.sh"; PATTERN="dataflex/weight_mean" ;;
  select)   SCRIPT="run_select_grpo.sh";   PATTERN="dataflex/kept_frac" ;;
  mix)      SCRIPT="run_mix_grpo.sh";       PATTERN="dataflex/prop_" ;;
  *) echo "unknown mechanism: $MECH (use reweight | select | mix)"; exit 2 ;;
esac

echo ">> Running DataFlex demo: $MECH"
echo ">> Script: examples/$SCRIPT   Log: $LOG"
echo ">> (cold start — Ray + vLLM load — takes a few minutes before step 1)"
echo

# stream to log; the mechanism scripts already print progress to stdout
bash "$HERE/$SCRIPT" 2>&1 | tee "$LOG"

echo
echo "================ DataFlex effect ($MECH) ================"
grep -oE "${PATTERN}[a-z0-9_]*:[0-9.]+" "$LOG" | tail -20 || \
  echo "(no '$PATTERN' metric found — check $LOG for errors)"
echo "========================================================="
