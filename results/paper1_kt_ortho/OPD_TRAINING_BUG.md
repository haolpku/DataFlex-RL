# OPD Joint Training: Ray Teacher-Pool Spawn Hang

**Date**: 2026-07-13.  
**Status**: Blocked. Escalated to future-work.

## Symptom

When launching a GRPO+OPD joint training run (student on 1 GPU, teacher pool on 2 GPUs, both Qwen-0.5B-Instruct), the training silently hangs at:

```
[TaskRunnerV1] INFO: actor and ref model engine initialized
[TaskRunnerV1] INFO: reward loop manager initialized
[WorkerDict] After FSDP, memory allocated: 5.23/95.00 GB
[Gloo] Rank 0 is connected to 0 peer ranks. Expected: 0
```

… and then nothing. No vLLM API server start line. No progress. GPU memory footprint stays under 5 GB (student only, teacher pool never loaded). No error message. Ray dashboard shows `TaskRunnerV1` alive but idle.

## What we know

- **The smoke test passed on 2026-07-12** with essentially the same configuration (student==teacher, both 0.5B, 4 GPUs, `distillation.n_gpus_per_node=2`). Reached step 2. `dataflex/weight_std=0.83`.
- **The same config fails on 2026-07-13** after ~14 h of the same session running (or fresh restarts). Every attempt hangs at the same spot.
- **Fresh Ray state clean does NOT help**: `pkill -9 -f 'ray|gcs|main_ppo|vllm'` + `rm -rf /tmp/ray*` + `rm -rf /dev/shm/ray*` between attempts — still hangs.
- **Reducing to 4 GPUs (CUDA_VISIBLE_DEVICES=0,1,2,3)** does not help.
- **Fresh clean slate before opd_s1** (no preceding baseline runs) still hangs.

## Hypotheses (ordered by likelihood)

1. **verl OPD's async agent loop deadlocks after `reward loop manager initialized`** on our specific Ray version. Not reproducible on the reference machine where smoke passed. May be a race between `TransferQueueController` and the teacher `AgentLoopWorker`.
2. **vLLM v1 API server / Ray placement group interaction**: teacher's `vLLMHttpServer` tries to grab GPU that's already promised to student's placement group, holding forever.
3. **CUDA shared-memory leak** across previous runs. Ray placement group can't book the resource. `nvidia-smi` shows 0 MiB free on non-student GPUs after clean, but no processes hold them — suggests kernel-level orphan.
4. **`distillation.enabled=true` config subtly differs from smoke**. Diffs checked: identical.

## What we tried

| Attempt | Result |
|---|---|
| Full pkill + `/tmp/ray*` clean between runs | Same hang |
| Reduce GPUs 8 → 4 | Same hang |
| Cold-start opd_s1 (no preceding baseline) | Same hang |
| Fix `_teacher_logp` shape (M1 bugfix from 2026-07-12) | Applies but doesn't affect init hang |
| Wait 14+ hours per attempt | No recovery |

## Options for future work

1. **Bisect the difference between the 2026-07-12 smoke run and now**: what changed? Ray version, vllm version, GPU driver, transformers version? Diff the environment. Most concrete lead.
2. **Try a different Ray version**: pin ray=2.44 or 2.50 (we're on 2.56). Some verl users report better OPD stability on 2.44.
3. **Use verl's `AsyncActorRolloutRefWorker`** instead of the default sync worker. Distillation was primarily developed for async; sync mode may have less-tested paths.
4. **Switch to a manual teacher-inference implementation**: skip verl's teacher pool. Run a standalone vLLM server for the teacher, hit its API from a custom scorer inside DataFlex. Higher engineering cost but avoids Ray placement group.
5. **Abandon joint RL+OPD** and use Paper 1's **offline orthogonality result** (Section `paper1_kt_ortho/README.md`) as the sole empirical evidence. Frame Paper 1 as "we propose the signal + prove orthogonality + rescue-ablation; joint training benchmark is future work due to infrastructure bugs".

We chose option 5 for the current milestone because Option B alone establishes the paper's core novelty (k_t ⊥ |advantage|) and running out of Alden's remaining time.

## Environment snapshot

```
verl               editable @ /apdcephfs_zwfy14/share_304380933/qifengcai/old/frameworks/verl
dataflex_verl      editable @ /apdcephfs_zwfy14/share_304380933/qifengcai/old/DataFlex-RL-opd  (feat/opd-fusion)
torch              2.7.1
ray                2.56.0
vllm               (v1, VLLM_USE_V1=1)
Qwen2.5-0.5B-Instruct  (student + teacher for self-teacher smoke)
CUDA               12.9
```

## For the successor

The MOST valuable thing to try first is option 1 (env bisect). If Ray/vllm version is pinned matching what verl OPD's authors tested, this likely works. Reference: verl `examples/on_policy_distillation_trainer/run_qwen3_0.6b_opd_veomni.sh` and its environment. If pinning env fixes it, then the 9 remaining runs (opd_s2/s3, opd_kt_select_{1,2,3}, opd_kt_reweight_{1,2,3}) can be run to complete Paper 1's MVE.
