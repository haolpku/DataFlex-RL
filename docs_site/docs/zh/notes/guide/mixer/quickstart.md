---
title: 快速开始
icon: solar:play-circle-outline
createTime: 2026/07/03 10:00:00
permalink: /zh/guide/mixer/quickstart/
---

# Mixer — 快速开始

**Mixer** 动态设置跨数据域的采样**配比**。与 Reweight/Select 不同,它作用在 **rollout 之前**、
**回顾性**地:在滑动窗口上累积每个域的平均 reward/advantage,调节*未来*采样。它需要 warmup
阶段(冷启动)和**多域数据集**。

## 数据:保留 `data_source`,加一个 `domain` 列

verl 用 `data_source` 选择 reward 函数,所以**不要覆盖它**。把域标签放在单独列(默认
`domain`)。有个 helper 按题目长度构建 2 域 GSM8K:

```bash
python examples/build_2domain_gsm8k.py --src $DATA/gsm8k --dst $DATA/gsm8k_2domain
```

## 通过配置启用

Mix 需要 trainer mode **和**自定义 replay-buffer 采样器:

```yaml
trainer:
  v1:
    trainer_mode: dataflex_mix_sync
    sampler:
      custom_sampler: {path: pkg://dataflex_verl.replay_buffer, name: DataFlexMixReplayBuffer}
dataflex:
  mechanism: mix
  domains: [gsm8k_short, gsm8k_long]   # 数据集 `domain` 列里的名字
  domain_key: domain                   # 存域标签的列(默认)
  scorer:   {name: reward_difficulty}
  actuator: {name: reward_gap, params: {temperature: 1.0, floor: 0.05}}
  warmup_step: 5
  update_step: 5
```

::: warning custom_sampler.path 要加 `pkg://`
用 `pkg://dataflex_verl.replay_buffer` —— 裸模块名会被 verl 当成文件路径,报 `FileNotFoundError`。
:::

可运行:`examples/run_mix_grpo.sh`。

## 看什么

```
dataflex/prop_gsm8k_short:0.5   dataflex/prop_gsm8k_long:0.5
dataflex/reward_gsm8k_short:... dataflex/reward_gsm8k_long:...
```

配比从均匀开始,随各域 reward 统计分化而偏向落后 / 更可学的域。短冒烟里两域 reward 可能仍为 0
(冷启动),配比保持均匀 —— 属正常。

## 可用的 mixer

| 名称 | 信号 | 规则 |
|---|---|---|
| `reward_gap` | per-domain 平均 reward | softmax over `max−reward`(偏向落后域) |
| `dump_ucb` | per-domain 平均 \|advantage\| | UCB + softmax(偏向可学 + 探索) |
| `static` | — | 固定配比(基线/warmup) |

::: tip Mix 什么时候有用?
2 个域时动态混合几乎没用(退化成一个比例旋钮)。它在**≥3 个难度异质的域**时才有价值,且
**可学习性**信号(|advantage|,`dump_ucb`)比原始 reward gap 更稳 —— 后者可能对不可解的域
过度投入。
:::
