---
title: 简介
icon: mdi:tooltip-text-outline
createTime: 2026/07/03 10:00:00
permalink: /zh/guide/basicinfo/intro/
---

# DataFlex-RL

**面向 LLM 强化学习微调的数据调度(选择 · 混合 · 重加权)—— 一个零侵入的
[verl](https://github.com/volcengine/verl) 插件。**

DataFlex-RL 把 DataFlex 的数据中心训练理念带到强化学习(PPO / GRPO / RLVR)。它挂载到
verl 的开放注册表,**不修改 verl 源码**:装好两个包,在 YAML 里加几行,用 verl 原生入口
运行即可。

```bash
pip install verl
pip install dataflex_verl
```

## 为什么 RL 需要数据调度?

在 RL 微调里,并非每条 rollout 都同样有用。一个 prompt 若所有回答都对(或都错)就没有
梯度信号;一个低概率 token 可能主导更新;某个数据域可能早已掌握而另一个还在落后。
DataFlex-RL 让模型**从训练中已经产生的信号出发,动态决定强调、丢弃或多采哪些数据** ——
无需第二个 reward model,也无需额外前向。

RL 场景其实比 SFT **更适合**做数据调度:除了 loss,每一步还产生 **reward、advantage、
per-token log-prob、以及分组结构(uid)** —— 一组强得多的可调度信号。

## 三种机制

| 机制 | 作用 | 挂载点 |
|---|---|---|
| **Reweight(重加权)** | per-token / per-sample 的 loss 权重 | trainer,advantage 之后 |
| **Select(选择)** | 丢弃样本(梯度置零) | trainer,advantage 之后 |
| **Mix(混合)** | 动态域采样配比 | 自定义 replay buffer,rollout 之前 |

三者用同一套设计表达:共享的 **Scorer**(`信号 → 分数`)喂给机制特定的 **Actuator**
(`分数 → 动作`)。设计理念见[框架设计](../framework/)。

## 与 DataFlex 的关系

[DataFlex](https://github.com/OpenDCAI/DataFlex) 是面向 **SFT** 的原始数据中心训练系统,
基于 LLaMA-Factory。DataFlex-RL 是它的 **RL 版本**:沿用同样的"选择/混合/重加权"词汇和
同样的 Scorer/Actuator 分解,但重新落在 verl 的 RL 训练循环与信号之上。二者共享设计基因,
但是独立的包(宿主框架不同)。

## 已验证

三种机制都能在 8×GPU 上端到端训练(Qwen2.5,GSM8K / dapo-math,verl v1),并通过
Qwen2.5-Math harness 评测。可运行的对比脚本与结果见仓库 `experiments/`。

## 下一步

- [框架设计](../framework/) —— Scorer / Actuator 架构与 verl 挂载点
- [安装](../install/) —— 安装 + 零配置自动注册
- 然后深入某个机制:[Reweighter](/zh/guide/reweighter/quickstart/) ·
  [Selector](/zh/guide/selector/quickstart/) · [Mixer](/zh/guide/mixer/quickstart/)
