# OPD Joint Training: Ray Teacher-Pool Spawn Hang

**Date last updated**: 2026-07-13.  
**Status**: Root cause **narrowed down**, quick fix not obvious.

## Symptom

GRPO+OPD training silently hangs at the same point every time:
- Log stops after `[TaskRunnerV1] reward loop manager initialized`.
- Only the student model loads (GPU 0 ~ 4.7 GB); the teacher pool never starts up.
- No error message. No progress. Waits forever.
- Reproducible across:
  - `multidomain_3` corpus **or** `dapo_math` corpus (dataset-independent);
  - `custom_reward_function` on **or** off;
  - 4 GPUs, 8 GPUs, 2 GPUs (resource-independent);
  - Fresh Ray state (`rm -rf /tmp/ray*`), fresh session, no preceding baseline runs.
- **Yet**: the same config succeeded on 2026-07-12, running through step 2 with `dataflex/weight_std=0.83`.

## Root cause: probable

Between the successful smoke on 2026-07-12 and the hangs on 2026-07-13, the trainer code path that instantiates the teacher pool has a **nested-asyncio conflict**:

- `TrainerBase._setup()` at `verl/trainer/ppo/v1/trainer_base.py:262` calls `MultiTeacherModelManager(...)` **directly**.
- `MultiTeacherModelManager.__init__` calls `_initialize_teacher_model_managers` → `_run_all(...)` which uses `asyncio.gather` under `@auto_await`.
- The `@auto_await` decorator falls back to `asyncio.run(...)` when no loop is running.
- The `fully_async_rollouter.py` code path around line 625 has an **explicit comment**:
  > "MultiTeacherModelManager.__init__ calls _run_all internally which uses asyncio.run(), conflicting with the already-running event loop. Run in a thread executor."
- The v1 trainer's `_setup()` is called inside a Ray actor method, which in vllm v1 mode runs with an active event loop → nested-asyncio → deadlock.

So the current code path only works in **sync-only** contexts (where no event loop is running at `_setup()` time). This may have been the case on 2026-07-12 (something in the environment activated a sync-only path), and something changed to trigger the async-context deadlock today.

## Confirmed by debug logging

With `RAY_LOG_TO_STDERR=1 RAY_BACKEND_LOG_LEVEL=debug`:
- GCS server starts fine.
- Actor placement group is scheduled for the student.
- After that, **no more placement groups are scheduled** — the teacher placement group request never fires from `MultiTeacherModelManager._initialize_llm_servers`.
- Consistent with the code deadlocking inside `_run_all(...)` in a nested-asyncio loop.

## Attempts that did not work

| # | Attempt | Result |
|---|---|---|
| 1 | Full pkill + `/tmp/ray*` + `/dev/shm/ray*` clean | Same hang |
| 2 | Reduce 8 → 4 GPUs (CUDA_VISIBLE_DEVICES=0,1,2,3) | Same hang |
| 3 | Reduce to 2 GPUs (student 1 + teacher 1) | Same hang |
| 4 | Cold-start opd_s1 (no preceding baseline runs) | Same hang |
| 5 | EXACT smoke config from 2026-07-12 (dapo_math, no custom_reward) | Same hang |
| 6 | Apply M1 shape fix for teacher_logprobs | Applies but doesn't affect init hang |
| 7 | Wait 14+ hours per attempt for a slow init | No recovery |

## Recommended fixes (in order of estimated cost)

### 1. Use `fully_async_rollouter` path (~1-2 h for someone who knows verl)

The comment in `fully_async_rollouter.py:625` explicitly says the correct pattern is to call `MultiTeacherModelManager.__init__` in a **thread executor**, not directly. The fix is to wrap the `MultiTeacherModelManager(...)` call at `trainer_base.py:262` similarly. Concretely, change:

```python
# trainer_base.py, around line 262:
self.teacher_model_manager = MultiTeacherModelManager(
    config=self.config,
    resource_pool=teacher_resource_pool,
)
```

to:

```python
import concurrent.futures
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
    fut = pool.submit(
        MultiTeacherModelManager,
        config=self.config,
        resource_pool=teacher_resource_pool,
    )
    self.teacher_model_manager = fut.result()
```

This avoids the nested-asyncio conflict.

### 2. Downgrade Ray to a version known-tested by verl (~1 h)

The verl OPD reference example uses Ray 2.44. We're on Ray 2.46. Some Ray versions behave differently around asyncio ↔ Ray actor interop. Try `pip install ray==2.44.0`.

### 3. Downgrade vLLM to v0 API (revert VLLM_USE_V1=1) (~30 min)

vLLM v1's async engine adds an event loop layer. Reverting to v0 (`export VLLM_USE_V1=0`) removes it and may make sync-only trainer code work.

### 4. Use `main_ppo_v0.py` entry point (~15 min try)

verl still ships the older `main_ppo_v0.py` entry which has a different distillation setup. Change `python -m verl.trainer.main_ppo` to `python -m verl.trainer.main_ppo_v0` and see if that path works. **Fastest thing to try first.**

## Environment snapshot

```
verl               editable @ /apdcephfs_zwfy14/share_304380933/qifengcai/old/frameworks/verl
dataflex_verl      editable @ /apdcephfs_zwfy14/share_304380933/qifengcai/old/DataFlex-RL-opd (feat/opd-fusion)
torch              2.7.1
ray                2.46.0
vllm               0.10.2 (v1 mode: VLLM_USE_V1=1)
Qwen2.5-0.5B-Instruct  (student + teacher)
CUDA               12.9
```

## For the successor

**First thing to try**: fix #4 (main_ppo_v0 entry). If that works, the 9 remaining OPD runs (`opd_{s1,s2,s3}`, `opd_kt_select_{s1,s2,s3}`, `opd_kt_reweight_{s1,s2,s3}`) can be completed in ~20 h wall-clock.

**Fallback**: Paper 1 stands on the current Option B evidence (see `results/paper1_kt_ortho/README.md`). k_t orthogonality is empirically established (mean r = +0.04 with 0.5B teacher, mean r = −0.05 with 7B teacher). Full RL+OPD scheduling benchmark is legitimate future work.
