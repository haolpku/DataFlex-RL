---
title: Quick Start
icon: solar:play-circle-outline
createTime: 2026/07/03 10:00:00
permalink: /en/guide/mixer/quickstart/
---

# Mixer — Quick Start

A **Mixer** dynamically sets the sampling **proportions** across data domains. Unlike
Reweight/Select, it acts **pre-rollout** and **retrospectively**: it accumulates each
domain's mean reward/advantage over a sliding window and steers *future* sampling. It
needs a warmup phase (cold start) and a **multi-domain dataset**.

## Data: keep `data_source`, add a `domain` column

verl uses `data_source` to pick the reward function, so **do not overwrite it**. Put
the domain label in a separate column (default `domain`). A helper builds a 2-domain
GSM8K split by question length:

```bash
python examples/build_2domain_gsm8k.py --src $DATA/gsm8k --dst $DATA/gsm8k_2domain
```

## Enable via config

Mix needs both a trainer mode **and** a custom replay-buffer sampler:

```yaml
trainer:
  v1:
    trainer_mode: dataflex_mix_sync
    sampler:
      custom_sampler: {path: pkg://dataflex_verl.replay_buffer, name: DataFlexMixReplayBuffer}
dataflex:
  mechanism: mix
  domains: [gsm8k_short, gsm8k_long]   # names in the dataset's `domain` column
  domain_key: domain                   # column holding the domain label (default)
  scorer:   {name: reward_difficulty}
  actuator: {name: reward_gap, params: {temperature: 1.0, floor: 0.05}}
  warmup_step: 5
  update_step: 5
```

::: warning custom_sampler.path needs `pkg://`
Use `pkg://dataflex_verl.replay_buffer` — a bare module name is treated as a file path
by verl and fails with `FileNotFoundError`.
:::

Runnable: `examples/run_mix_grpo.sh`.

## What to look for

```
dataflex/prop_gsm8k_short:0.5   dataflex/prop_gsm8k_long:0.5
dataflex/reward_gsm8k_short:... dataflex/reward_gsm8k_long:...
```

Proportions start uniform and shift toward lagging / more-learnable domains as their
reward stats diverge. In a short smoke both rewards may still be 0 (cold start), so
proportions stay uniform — expected.

## Available mixers

| Name | Signal | Rule |
|---|---|---|
| `reward_gap` | per-domain mean reward | softmax over `max−reward` (favor lagging) |
| `dump_ucb` | per-domain mean \|advantage\| | UCB + softmax (favor learnable + explore) |
| `static` | — | fixed proportions (baseline/warmup) |

::: tip When does Mix help?
Dynamic mixing barely helps with **2 domains** (reduces to one ratio knob). It pays off
at **≥3 domains with heterogeneous difficulty**, and a **learnability** signal
(|advantage|, `dump_ucb`) is safer than raw reward gap, which can over-invest in
unsolvable domains.
:::
