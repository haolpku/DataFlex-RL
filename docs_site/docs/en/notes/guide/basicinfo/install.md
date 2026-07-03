---
title: Installation
icon: material-symbols:download
createTime: 2026/07/03 10:00:00
permalink: /en/guide/basicinfo/install/
---

# Installation

## Install

```bash
pip install verl          # host RL framework (v1 trainer; use_v1=true by default)
pip install dataflex_verl # this plugin
```

From source (development):

```bash
git clone https://github.com/haolpku/DataFlex-RL.git
cd DataFlex-RL
pip install -e ".[dev]"
```

## Sanity check (no GPU needed)

```bash
# framework-agnostic unit tests: scorers, actuators, registry, compat checks
pytest -q

# zero-config autoload: merely importing verl registers our trainers
python -c "
import verl
from verl.trainer.ppo.v1.trainer_base import TRAINER_REGISTRY
assert {'dataflex_sync', 'dataflex_mix_sync'} <= set(TRAINER_REGISTRY)
print('OK:', sorted(TRAINER_REGISTRY))
"
```

If the two `dataflex_*` trainers appear, verl's entry-point plugin discovery is
working — you never import `dataflex_verl` yourself, and neither do verl's Ray workers.

## Requirements / compatibility

- **verl with the v1 trainer** (`config.trainer.use_v1=true`, the default), which uses
  the TransferQueue data plane and the `register_trainer` / custom-sampler hooks.
- **Reweight / Select** need an advantage estimator that populates standard batch
  fields (GRPO, GAE, RLOO, …). Group scorers (`needs_groups=True`) require a
  group-based estimator (GRPO/RLOO/…).
- No flash-attn required: examples use `attn_implementation=sdpa` +
  `use_remove_padding=False`.

## Scaling notes

- **GPUs:** set `CUDA_VISIBLE_DEVICES` and `trainer.n_gpus_per_node`.
- **Ray CPUs:** on a many-CPU box, set `ray_kwargs.ray_init.num_cpus` to ~8×(#GPUs).
  Too few CPU slots and Ray deadlocks creating the colocated worker groups (symptom:
  log stalls at `create worker group`, GPUs idle).
- **Cold start** (Ray + vLLM engine + CUDA-graph capture) takes several minutes before
  step 1 regardless of GPU count — normal.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Unknown trainer 'dataflex_sync'` | plugin not discovered — reinstall; ensure `VERL_USE_EXTERNAL_PLUGINS` isn't `none`. |
| Hang at `create worker group`, GPUs idle | raise `ray_kwargs.ray_init.num_cpus` (~8×#GPUs). |
| `FlashAttention2 ... not installed` | set `+actor_rollout_ref.model.override_config.attn_implementation=sdpa` and `use_remove_padding=False`. |
| `NotImplementedError: Reward function ... data_source=...` (mix) | don't overwrite `data_source`; put the domain in a separate column and set `dataflex.domain_key`. |
| `FileNotFoundError: Custom module file not found` (mix) | `custom_sampler.path` needs the `pkg://` prefix: `pkg://dataflex_verl.replay_buffer`. |
