---
title: 快速开始
icon: solar:play-circle-outline
createTime: 2026/07/03 10:00:00
permalink: /zh/guide/reweighter/quickstart/
---

# Reweighter — 快速开始

**Reweighter** 把 per-sample 或 per-token 的分数转成权重,在聚合前乘进策略梯度 loss
(`pg_losses`)。它复用 verl 现成的 `rollout_is_weights` 钩子 —— **无需自定义 policy loss**。
权重会归一化到均值 1,以保持等效学习率。

## 通过配置启用

在 verl 的 YAML 里加 `dataflex` 块(或用 `+dataflex.*` 命令行覆盖),并设置 trainer mode:

```yaml
trainer: {v1: {trainer_mode: dataflex_sync}}
dataflex:
  mechanism: reweight
  scorer:   {name: advantage_magnitude, params: {agg: mean}}
  actuator: {name: softmax, params: {temperature: 1.0}}
  warmup_step: 0
```

## 运行

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    ... (你平常的 verl 参数) ... \
    trainer.v1.trainer_mode=dataflex_sync \
    +dataflex.mechanism=reweight \
    +dataflex.scorer.name=advantage_magnitude \
    +dataflex.actuator.name=softmax \
    +dataflex.actuator.params.temperature=1.0 \
    +dataflex.warmup_step=0
```

可运行脚本见 `examples/run_reweight_grpo.sh`。

## 看什么

每步控制台日志里:

```
dataflex/weight_mean:1.0   dataflex/weight_std:...
```

`weight_mean ≈ 1.0` 说明权重已归一化并生效。

## 可用的 reweighter

| 名称 | 信号(scorer) | 规则 |
|---|---|---|
| `softmax` | 任意 per-sample 分数 | `softmax(score/T)`,强调高分 |
| `difficulty_band` | per-seq reward | 提升中等难度分位段 |
| `advantage_reweight` | per-token `token_prob` | `w=α·π_θ+(1−α)`,压低低概率 token(arXiv:2505.12929) |

见各算法页,以及[添加 Reweighter](../tutorial/) 编写自己的。
