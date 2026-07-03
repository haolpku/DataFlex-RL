---
title: Online Difficulty Filtering
icon: solar:filter-bold
createTime: 2026/07/03 10:00:00
permalink: /zh/guide/selector/difficulty_filtering/
---

# Online Difficulty Filtering(在线难度过滤)

**来源:** *Online Difficulty Filtering for Reasoning-Oriented RL*
([arXiv:2504.03380](https://arxiv.org/abs/2504.03380))。

## 动机

RL 的学习信号在 prompt 解出率接近 **0.5** 时最大:一个模型总能解出(rate=1)或从不解出
(rate=0)的 prompt,组 advantage 为零,不贡献梯度。Online Difficulty Filtering 保留当前
解出率落在 0.5 附近**平衡带**内的 prompt,把更新集中在最有信息量的地方。

## 规则

保留 prompt *x*,当且仅当其 batch 估计的解出率满足:

$$
T_\text{low} \le p(x) \le T_\text{high}
$$

例如 `T_low = 0.2, T_high = 0.8`。这**推广了 DAPO dynamic sampling** —— 后者是退化带
`(0, 1)`,只排除恰好 0 和恰好 1。

## 在 DataFlex-RL 中

无需新代码 —— 就是 `group_solve_rate` scorer + `threshold_band` selector 配 0.2/0.8 带宽:

```yaml
trainer: {v1: {trainer_mode: dataflex_sync}}
dataflex:
  mechanism: select
  scorer:   {name: group_solve_rate, params: {success_threshold: 0.5}}
  actuator: {name: threshold_band, params: {low: 0.2, high: 0.8}}
  warmup_step: 0
```

## 证据与条件

报告相对 GRPO **+10% AIME / +4% 平均**(五个数学 benchmark);机制与规模无关。**重要条件:**
提升需要数据集有**较宽的难度分布**。在难度均匀偏易的集合(如纯 GSM8K)上可过滤的东西很少,
过度过滤反而会削薄有效 batch、**损害**性能 —— 因此应搭配更难 / 更多样的数据(如 dapo-math)。

## 信号 / 规则 / 粒度

| 属性 | 值 |
|---|---|
| 信号 | per-group 解出率(来自 `rm_scores` + `uid`) |
| 规则 | 保留解出率 ∈ [0.2, 0.8] |
| 粒度 | prompt(组) |
| 需要分组 | 是(GRPO/RLOO/…) |
| 额外前向 | 无 |
