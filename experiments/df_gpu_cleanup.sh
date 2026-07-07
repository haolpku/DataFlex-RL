#!/usr/bin/env bash
# Aggressively free all GPU memory held by a (possibly crashed) verl/vllm run in THIS
# namespace. Call between runs. Safe to source or run. Only touches our own training
# procs (ray, vllm engine cores, main_ppo, FSDP workers) — never other containers.
df_gpu_cleanup() {
  ray stop --force >/dev/null 2>&1 || true
  pkill -9 -f "main_ppo"            2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore"    2>/dev/null || true
  pkill -9 -f "EngineCore_DP"       2>/dev/null || true
  pkill -9 -f "from multiprocessing.spawn" 2>/dev/null || true
  pkill -9 -f "vllm.v1.engine"      2>/dev/null || true
  pkill -9 -f "raylet"              2>/dev/null || true
  pkill -9 -f "ray::"               2>/dev/null || true
  pkill -9 -f "WorkerDict"          2>/dev/null || true
  pkill -9 -f "AgentLoopWorker"     2>/dev/null || true
  sleep 8
}
# if executed directly (not sourced), run it
[ "${BASH_SOURCE[0]}" = "$0" ] && df_gpu_cleanup
