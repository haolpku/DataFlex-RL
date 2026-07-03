---
title: 快速开始
icon: solar:play-circle-outline
createTime: 2026/07/03 10:00:00
permalink: /zh/guide/selector/quickstart/
---

# Selector — 快速开始

**Selector** 把 per-sample 或 per-group 分数转成一组保留索引;被丢弃的样本梯度为零。在 RL 中
这会在 rollout *之后*移除样本对更新的贡献。

::: tip 权重 0 ≠ 省下 rollout 成本
在这里(advantage 之后)选择会移除梯度,但**不**省 rollout —— 生成已经发生。真正*省 rollout*
的变体在 replay-buffer / rollout 前的层。
:::

## 通过配置启用

```yaml
trainer: {v1: {trainer_mode: dataflex_sync}}
dataflex:
  mechanism: select
  scorer:   {name: group_solve_rate, params: {success_threshold: 0.5}}
  actuator: {name: threshold_band, params: {low: 0.0, high: 1.0}}
```

## 运行

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    ... (平常的 verl 参数) ... \
    trainer.v1.trainer_mode=dataflex_sync \
    +dataflex.mechanism=select \
    +dataflex.scorer.name=group_solve_rate \
    +dataflex.scorer.params.success_threshold=0.5 \
    +dataflex.actuator.name=threshold_band \
    +dataflex.actuator.params.low=0.0 \
    +dataflex.actuator.params.high=1.0
```

可运行:`examples/run_select_grpo.sh`。

## 看什么

```
dataflex/kept_frac:0.078...
```

过滤后保留样本的比例。

## 可用的 selector

| 名称 | 信号(scorer) | 规则 |
|---|---|---|
| `threshold_band` | `group_solve_rate` | 保留分数 ∈ (low, high) |
| `topk_fraction` | `advantage_magnitude` | 保留 top/bottom 比例 |

**DAPO dynamic sampling** = `group_solve_rate` + `threshold_band(0,1)`(丢掉全对/全错的组)。
**Online Difficulty Filtering** = 同样但带宽 `(0.2, 0.8)` —— 见其[算法页](../difficulty_filtering/)。
