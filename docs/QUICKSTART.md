# Quickstart

A copy-paste walkthrough: install, sanity-check, and run each of the three
mechanisms (reweight / select / mix) end-to-end on GRPO.

Everything here was verified on **8× H20, Qwen2.5-0.5B, GSM8K, verl v1**.

---

## 1. Install

```bash
pip install verl          # the host RL framework (v1 trainer; use_v1=true by default)
pip install dataflex_verl # this package
```

Or from source for development:

```bash
git clone https://github.com/haolpku/DataFlex-RL.git
cd DataFlex-RL
pip install -e ".[dev]"
```

## 2. Sanity check (no GPU needed)

Confirm the framework-agnostic core works and the plugin auto-registers into verl.

```bash
# 24 offline unit tests — scorers, actuators, registry, compat checks
pytest -q

# zero-config autoload: merely importing verl registers our trainers
python -c "
import verl
from verl.trainer.ppo.v1.trainer_base import TRAINER_REGISTRY
assert {'dataflex_sync', 'dataflex_mix_sync'} <= set(TRAINER_REGISTRY)
print('OK — dataflex trainers auto-registered:', sorted(TRAINER_REGISTRY))
"
```

If the second command prints the two `dataflex_*` trainers, the entry-point plugin
discovery is working — you never import `dataflex_verl` yourself, and neither do
verl's Ray workers.

## 3. Point the demos at your model & data

The `examples/*.sh` scripts default to:

```
ROOT=/apdcephfs_zwfy14/share_304380933/aldenliang
MODEL=$ROOT/models/Qwen2.5-0.5B-Instruct
DATA=$ROOT/data/gsm8k
```

Edit `ROOT`/`MODEL`/`DATA` at the top of each script for your environment. Any
GRPO-ready GSM8K parquet (verl's standard format) works.

## 4. Run a mechanism

Each script is a stock `python -m verl.trainer.main_ppo ...` invocation plus a
`config.dataflex` block. verl starts Ray itself — **do not** `ray start` manually.

### Reweight — emphasize high-|advantage| samples
```bash
bash examples/run_reweight_grpo.sh
```
Look for, in the per-step log line:
```
dataflex/weight_mean:1.0   dataflex/weight_std:...
```
Per-sample weights (softmax over |advantage|, mean-normalized to 1) are multiplied
into `pg_losses` via `rollout_is_weights`.

### Select — DAPO-style dynamic sampling
```bash
bash examples/run_select_grpo.sh
```
Look for:
```
dataflex/kept_frac:0.078...
```
All-solved (rate=1) and all-failed (rate=0) GRPO groups are dropped; only groups
with a nonzero learning signal keep their gradient contribution.

### Mix — dynamic domain proportions
Mix needs ≥2 domains. Build a demo 2-domain split of GSM8K first (keeps the real
`data_source` so verl's reward still resolves; adds a `domain` column):
```bash
python examples/build_2domain_gsm8k.py \
    --src $ROOT/data/gsm8k --dst $ROOT/data/gsm8k_2domain
bash examples/run_mix_grpo.sh
```
Look for:
```
dataflex/prop_gsm8k_short:0.5   dataflex/prop_gsm8k_long:0.5
dataflex/reward_gsm8k_short:...  dataflex/reward_gsm8k_long:...
```
Proportions start uniform and shift toward the lagging (lower-reward) domain as
each domain's sliding-window reward diverges. In a 2–4 step smoke both rewards are
still 0 (cold start), so proportions stay 0.5/0.5 — expected; they move once a
domain starts solving.

## 5. One-command demo

`examples/run_demo.sh` runs a chosen mechanism and greps out the DataFlex metric so
you see the effect without scrolling the log:

```bash
bash examples/run_demo.sh reweight   # or: select | mix
```

## Scaling notes

- **GPUs**: set `CUDA_VISIBLE_DEVICES` and `trainer.n_gpus_per_node`. The demos use 8.
- **Ray CPUs**: on a many-CPU box, set `ray_kwargs.ray_init.num_cpus` to ~8×(#GPUs).
  Too few CPU slots and Ray deadlocks while creating the colocated worker groups
  (symptom: the log stalls at `create worker group` with GPUs idle). The demos use 64.
- **Cold start** (Ray + vLLM engine + CUDA-graph capture) takes several minutes
  before step 1 regardless of GPU count — normal.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Unknown trainer 'dataflex_sync'` | plugin not discovered. Reinstall `dataflex_verl`; verify step 2's autoload check. Ensure `VERL_USE_EXTERNAL_PLUGINS` isn't set to `none`. |
| Hang at `create worker group`, GPUs idle | raise `ray_kwargs.ray_init.num_cpus` (~8×#GPUs). |
| `NotImplementedError: Reward function ... data_source=...` (mix) | don't overwrite `data_source`; put the domain in a separate column and set `dataflex.domain_key`. Use `build_2domain_gsm8k.py`. |
| `FileNotFoundError: Custom module file not found` (mix) | `custom_sampler.path` needs the `pkg://` prefix: `pkg://dataflex_verl.replay_buffer`. |
| Scorer rejected at startup | a `needs_groups` scorer (e.g. `group_solve_rate`) requires a group estimator (GRPO/RLOO/…), not GAE. |
