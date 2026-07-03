---
title: Quick Start
icon: solar:play-circle-outline
createTime: 2026/07/03 10:00:00
permalink: /en/guide/selector/quickstart/
---

# Selector — Quick Start

A **Selector** turns a per-sample or per-group score into a set of kept indices;
dropped samples contribute zero gradient. In RL this removes a sample's contribution
to the update *after* rollout.

::: tip weight 0 ≠ dropping rollout cost
Selecting here (post-advantage) removes the gradient but does **not** save the rollout —
generation already happened. The rollout-*saving* variant lives at the replay-buffer /
pre-rollout layer.
:::

## Enable via config

```yaml
trainer: {v1: {trainer_mode: dataflex_sync}}
dataflex:
  mechanism: select
  scorer:   {name: group_solve_rate, params: {success_threshold: 0.5}}
  actuator: {name: threshold_band, params: {low: 0.0, high: 1.0}}
```

## Run

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    ... (usual verl args) ... \
    trainer.v1.trainer_mode=dataflex_sync \
    +dataflex.mechanism=select \
    +dataflex.scorer.name=group_solve_rate \
    +dataflex.scorer.params.success_threshold=0.5 \
    +dataflex.actuator.name=threshold_band \
    +dataflex.actuator.params.low=0.0 \
    +dataflex.actuator.params.high=1.0
```

Runnable: `examples/run_select_grpo.sh`.

## What to look for

```
dataflex/kept_frac:0.078...
```

the fraction of samples kept after filtering.

## Available selectors

| Name | Signal (scorer) | Rule |
|---|---|---|
| `threshold_band` | `group_solve_rate` | keep score ∈ (low, high) |
| `topk_fraction` | `advantage_magnitude` | keep top/bottom fraction |

**DAPO dynamic sampling** = `group_solve_rate` + `threshold_band(0,1)` (drop all-solved
/ all-failed groups). **Online Difficulty Filtering** = the same with band `(0.2, 0.8)`
— see its [algorithm page](../difficulty_filtering/).
