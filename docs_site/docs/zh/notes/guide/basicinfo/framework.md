---
title: 框架设计
icon: material-symbols:auto-transmission-sharp
createTime: 2026/07/03 10:00:00
permalink: /zh/guide/basicinfo/framework/
---

# 框架设计

## 概述

DataFlex-RL 是 [verl](https://github.com/volcengine/verl) RL 训练库的一个**零侵入插件**。
它在 RL 微调过程中智能调度训练数据,支持**动态样本重加权**、**样本选择**、**域配比调整**,
以提升训练效率和最终模型性能。

### 设计理念

核心理念是 **RL 的数据中心智能调度**:不再对每条 rollout 一视同仁,而是让模型根据训练中
已经产生的信号(reward、advantage、log-prob、分组结构)动态调整每个样本的贡献方式。
关键在于,DataFlex-RL **不修改 verl 源码**、**不做额外前向** —— 它读取的都是 RL 训练本就
会计算的 batch 字段。

## 核心架构

### 两层解耦:Scorer + Actuator

核心思想是把每个数据调度算法拆成两个可复用的层:

```
┌──────────────────────────────────────────────────────────────────────┐
│                         verl RL 训练循环                                │
│           rollout → reward → advantage → actor/critic 更新              │
├──────────────────────────────────────────────────────────────────────┤
│  DataFlex-RL 插件(注册进 verl 的开放注册表)                            │
│                                                                        │
│   ┌─────────────────────────────────────────────────────────────┐    │
│   │  Scorer   (信号 → 分数)   —  三机制共享                         │    │
│   │  只读声明的 batch 字段(reward / advantage / uid / log_prob)   │    │
│   │  返回 per-sample 或 per-token 的分数                            │    │
│   └─────────────────────────────────────────────────────────────┘    │
│                                │                                       │
│         ┌──────────────────────┼──────────────────────┐               │
│         ▼                      ▼                      ▼               │
│   ┌───────────┐          ┌───────────┐          ┌───────────┐         │
│   │ Reweighter│          │ Selector  │          │  Mixer    │         │
│   │ 分数→权重  │          │ 分数→保留  │          │ 分数→域配比│         │
│   └───────────┘          └───────────┘          └───────────┘         │
│  (per-token loss)      (丢弃样本)         (未来采样)                   │
└──────────────────────────────────────────────────────────────────────┘
```

**为什么这样拆?** 因为一个数据调度算法有两个独立的问题:
1. *基于什么信号打分?*(reward、advantage、组解出率、token 概率……)
2. *拿分数做什么?*(加权 loss、丢弃样本、重新配比域)

同一个 **Scorer** 喂给三种机制,所以打分逻辑**只写一次**,不必按机制或按 RL 算法重复。

### Scorer

Scorer 把训练信号映射为分数。它**声明自己的依赖**,以便宿主在挂载时**一次性**校验与当前
算法的兼容性:

```python
class Scorer(ABC):
    requires: list[str]      # 读取的 batch 字段,如 ["advantages", "response_mask"]
    timing: str              # "pre_rollout" | "post_reward" | "post_advantage" | "in_loss"
    granularity: str         # "domain" | "prompt" | "response" | "token"
    needs_groups: bool       # True → 仅适用于分组估计器(GRPO/RLOO)

    def score(self, batch, step_id, **ctx): ...   # → (bs,) 或 (bs, L)
```

`needs_groups` 是**不按算法分叉**的关键:分组专用的 scorer(如组解出率)在 PPO+GAE 下会被
启动时拒绝,而 reward/advantage 级的 scorer 对任何估计器都适用。一次配置检查替代 N 份按
算法复制的代码。

### Actuator

三种机制消费分数。它们在**挂载点**、**输出类型**、**成本语义**上不同 —— 这正是不能合并成
一个接口的原因:

| 机制 | 输出 | 挂载点 | 时机 |
|---|---|---|---|
| **Reweight** | per-token 权重 → `rollout_is_weights` | trainer `_compute_advantage` | 每步,循环内 |
| **Select** | 0/1 掩码(梯度置零) | trainer `_compute_advantage` | 每步,循环内 |
| **Mix** | 域配比 → 采样器 | 自定义 `ReplayBuffer` + trainer | 周期性,rollout 前 |

**一个 RL 特有的关键区分:**"权重 0"(Reweight)≠"丢弃"(Select)。被置零的样本仍然
付出了 rollout 代价;只有 *rollout 前*的选择才真正省下生成。Reweight 和 Select 共享
advantage 后的挂载点,因为两者都归结为 per-token 权重,而 verl 的原生 policy loss 本来就会
把它乘进去 —— **无需自定义 policy loss**。Mix 是回顾性、按域的:它累积每个域的平均 reward,
调节*未来*的采样,因此需要一个 warmup 阶段。

## 如何挂载进 verl(零侵入)

verl v1 暴露了开放注册表,DataFlex-RL 在 `import` 时注册进去:

1. **`register_trainer`** → 由 `config.trainer.v1.trainer_mode` 选择
   (reweight/select 用 `dataflex_sync`,mix 用 `dataflex_mix_sync`)。
2. **自定义 replay-buffer 采样器** → `config.trainer.v1.sampler.custom_sampler`
   (用于 Mix 的按域配比采样)。
3. **`rollout_is_weights`** → verl 现成的 per-token 权重钩子,vanilla PPO loss 会在聚合前
   把它乘进 `pg_losses`(Reweight 和 Select 写的就是它)。

### 零配置自动注册

因为 verl 在 `import verl` 时会扫描 `verl.plugins` entry-point 组,所以只要装了
`dataflex_verl`(其 entry point 指向 `dataflex_verl.autoload` 模块),就会在**主进程和每个
Ray worker 里**注册 trainer —— 无需手动 import,无需 `PYTHONPATH`。在 verl 的 YAML 里加一个
`config.dataflex` 块,跑原生入口即可。

## 组件层次

1. **宿主层(verl)** —— rollout、reward、advantage、actor/critic worker、TransferQueue 数据面。
2. **Trainer 层(DataFlex-RL trainer)** —— 继承 verl v1 sync trainer;override
   `_compute_advantage`(reweight/select)或采样器 + 统计钩子(mix)。不重写 `fit()`。
3. **组件层** —— `Scorer` + `Actuator`(Reweighter / Selector / Mixer)。
4. **注册表** —— 框架无关的 `core/`:注册、配置构建、兼容性校验。

**关键特性:** DataFlex-RL 不在 verl 之上加层,而是**注册进 verl 已有的钩子**,保持 RL 循环
不变的同时增强数据调度。

## 时序模型:SFT 离散相位 vs RL 循环内

一个常见误解是把 RL 数据调度当成 SFT 那种"选一批 → 训 → 再选"的离散相位循环。**并非如此:**

- **Reweight / Select** 是**循环内**的:每一步,在 reward/advantage 之后,scorer 读当前
  batch,actuator 立即作用其上。
- **Mix** 是唯一保持周期节奏的机制 —— 它在滑动窗口上累积每个域的统计,更新未来采样配比
  (回顾性,需要 warmup 冷启动)。

## 下一步

- [安装](../install/)
- [Reweighter](/zh/guide/reweighter/quickstart/) · [Selector](/zh/guide/selector/quickstart/) · [Mixer](/zh/guide/mixer/quickstart/)
- 要添加自己的算法,见各机制的 **tutorial** 页。
