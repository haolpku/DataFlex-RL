---
title: DUMP(UCB Mixer)
icon: solar:chart-2-outline
createTime: 2026/07/03 10:00:00
permalink: /zh/guide/mixer/dump_ucb/
---

# DUMP — UCB Bandit Mixer

**来源:** *DUMP: Automated Distribution-Level Curriculum Learning for RL-based LLM
Post-training* ([arXiv:2504.09710](https://arxiv.org/abs/2504.09710))。

## 动机

默认的 `reward_gap` mixer 是**纯利用**的:它总是偏向 reward 最低的域,这可能对一个单纯太难
(不可解 → 无法学习)的域过度投入。DUMP 把每个域当作 **bandit 臂**,其价值是**可学习性** ——
平均绝对 advantage `E[|Â|]`,并给欠采样的域加 UCB **探索项**。`|advantage|` 自调节:它在中等
难度时最大,在已掌握和不可解的域都接近零。

## 规则

对每个域 *d*,平均 |advantage| 为 `v_d`,访问次数 `n_d`(总数 `N`):

$$
\text{UCB}(d) = v_d + c\sqrt{\frac{2\ln(N+1)}{n_d + 1}}, \qquad
p_d = \mathrm{softmax}\big(\text{UCB}(d) / T\big)
$$

带一个 floor 避免饿死任何域。第一项利用高可学习性的域,第二项探索少采样的域。

## 在 DataFlex-RL 中

```yaml
trainer:
  v1:
    trainer_mode: dataflex_mix_sync
    sampler:
      custom_sampler: {path: pkg://dataflex_verl.replay_buffer, name: DataFlexMixReplayBuffer}
dataflex:
  mechanism: mix
  domains: [math, code, logic]
  scorer:   {name: advantage_magnitude}      # per-domain |advantage|
  actuator: {name: dump_ucb, params: {temperature: 1.0, c: 1.0, floor: 0.05}}
  warmup_step: 5
  update_step: 5
```

## 证据与条件

DUMP 在多域逻辑/数学上优于均匀采样,并自动诱导 easy→hard 课程。由于 `|advantage|` 信号正是
GRPO 已经算出的,它**几乎零开销**。与所有 mixer 一样,它需要**≥3 个异质域**才能体现价值;
2 个域时一个调好的静态比例就有竞争力。

## 信号 / 规则 / 粒度

| 属性 | 值 |
|---|---|
| 信号 | per-domain 平均 \|advantage\| + 访问次数 |
| 规则 | 域上的 UCB + softmax |
| 粒度 | domain |
| 需要分组 | 否(任何产出 advantage 的估计器) |
| 额外前向 | 无 |

## 值得添加的相关 mixer

调研(仓库 `research/03_mixture.md`)标出 **TSCL**(reward 斜率 / 学习进度)、**ODM/EXP3**
(对抗 bandit)、**VCRL**(组 reward 方差)为下一批候选。一个干净的加法是**可插拔 Mixer 后端**
—— 在发出 `{score, count, staleness}` 的 Scorer 之上,softmax / UCB / EXP3 / Thompson 变成可
互换的策略。
