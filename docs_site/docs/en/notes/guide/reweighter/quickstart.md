---
title: Quick Start
icon: solar:play-circle-outline
createTime: 2026/07/03 10:00:00
permalink: /en/guide/reweighter/quickstart/
---

# Reweighter — Quick Start

A **Reweighter** turns a per-sample or per-token score into weights that multiply into
the policy-gradient loss (`pg_losses`) before aggregation. It reuses verl's existing
`rollout_is_weights` hook — **no custom policy loss needed**. Weights are
mean-normalized to 1 to preserve the effective learning rate.

## Enable via config

Add a `dataflex` block to verl's YAML (or `+dataflex.*` CLI overrides) and set the
trainer mode:

```yaml
trainer: {v1: {trainer_mode: dataflex_sync}}
dataflex:
  mechanism: reweight
  scorer:   {name: advantage_magnitude, params: {agg: mean}}
  actuator: {name: softmax, params: {temperature: 1.0}}
  warmup_step: 0
```

## Run

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    ... (your usual verl args) ... \
    trainer.v1.trainer_mode=dataflex_sync \
    +dataflex.mechanism=reweight \
    +dataflex.scorer.name=advantage_magnitude \
    +dataflex.actuator.name=softmax \
    +dataflex.actuator.params.temperature=1.0 \
    +dataflex.warmup_step=0
```

A runnable end-to-end script is in `examples/run_reweight_grpo.sh`.

## What to look for

In the per-step console log:

```
dataflex/weight_mean:1.0   dataflex/weight_std:...
```

`weight_mean ≈ 1.0` confirms the weights are mean-normalized and being applied.

## Available reweighters

| Name | Signal (scorer) | Rule |
|---|---|---|
| `softmax` | any per-sample score | `softmax(score/T)`, emphasize high-score |
| `difficulty_band` | per-seq reward | up-weight mid-difficulty quantile band |
| `advantage_reweight` | per-token `token_prob` | `w=α·π_θ+(1−α)`, damp low-prob tokens (arXiv:2505.12929) |

See individual algorithm pages, and [Add a Reweighter](../tutorial/) to write your own.
