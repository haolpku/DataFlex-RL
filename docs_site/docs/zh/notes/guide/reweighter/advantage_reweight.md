---
title: Advantage Reweighting
icon: material-symbols:balance
createTime: 2026/07/03 10:00:00
permalink: /zh/guide/reweighter/advantage_reweight/
---

# Advantage Reweighting(低概率 token 抑制)

**来源:** *Do Not Let Low-Probability Tokens Over-Dominate in RL for LLMs*
([arXiv:2505.12929](https://arxiv.org/abs/2505.12929))。

## 动机

在策略梯度 RL 里,一个 token 的梯度幅度与其概率成反比。低概率 token 因此产生**过大的梯度**,
可能主导更新、破坏稳定性。Advantage Reweighting(AR)用一个随 token 概率增大的 per-token
权重来抑制它们。

## 规则

对每个 token *t*,概率为 `π_θ(t)`:

$$
w_t = \alpha \cdot \pi_\theta(t) + (1 - \alpha)
$$

- 低概率 token(`π→0`)权重接近 `1−α`(被抑制)。
- 高概率 token(`π→1`)权重接近 `1`。
- 权重在有效 token 上归一化到均值 1,保持等效学习率。

`α ∈ [0, 1]` 控制抑制强度;`α=0` 退化为无权重基线。

## 在 DataFlex-RL 中

AR 是 **token 粒度**:用 `token_prob` scorer(per-token `π_θ = exp(old_log_prob)`,无需额外
前向)和 `advantage_reweight` reweighter(直接返回 `(bs, L)` 权重矩阵)。

```yaml
trainer: {v1: {trainer_mode: dataflex_sync}}
dataflex:
  mechanism: reweight
  scorer:   {name: token_prob}
  actuator: {name: advantage_reweight, params: {alpha: 0.5}}
  warmup_step: 0
```

命令行:

```bash
+dataflex.mechanism=reweight \
+dataflex.scorer.name=token_prob \
+dataflex.actuator.name=advantage_reweight \
+dataflex.actuator.params.alpha=0.5
```

## 为什么它是小模型的强选择

在调研的重加权方法里,AR 有**最直接的小模型证据**:论文在 **3B 和 7B** 上报告提升(如
Knights-and-Knaves 逻辑 +46% 相对,数学正向),且只需 `old_log_probs` —— 无额外前向、无
reward model。提升往往在**训练后期**变大。

## 信号 / 规则 / 粒度

| 属性 | 值 |
|---|---|
| 信号 | per-token `π_θ`(来自 `old_log_probs`) |
| 规则 | `w = α·π + (1−α)`,归一化 |
| 粒度 | token |
| 需要分组 | 否(任意 advantage 估计器) |
| 额外前向 | 无 |
