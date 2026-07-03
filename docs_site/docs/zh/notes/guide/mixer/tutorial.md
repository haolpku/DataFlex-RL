---
title: 添加 Mixer
icon: solar:add-circle-outline
createTime: 2026/07/03 10:00:00
permalink: /zh/guide/mixer/tutorial/
---

# 向 DataFlex-RL 添加 Mixer

Mixer 消费**per-domain 统计**,返回采样**配比**。与 Reweighter/Selector 不同,它不作用于当前
batch —— 它调节未来采样。

## 第 1 步:添加 Mixer

`act` 收到一个 `{域: 分数}` 字典(来自 trainer 的滑动窗口 tracker),返回与 `self.domains`
对齐的配比向量。

```python
import numpy as np
from .core.actuator import Mixer
from .core.registry import register_mixer

@register_mixer("reward_gap")
class RewardGapMixer(Mixer):
    """配比 ∝ softmax((max_reward − domain_reward) / T):偏向落后域。"""
    def __init__(self, domains, temperature=1.0, floor=0.05, **kw):
        super().__init__(**kw)
        self.domains = list(domains); self.temperature = temperature; self.floor = floor

    def act(self, scores, batch, **ctx) -> np.ndarray:
        means = np.array([scores.get(d, 0.0) for d in self.domains])
        z = (means.max() - means) / max(self.temperature, 1e-6)
        z -= z.max()
        p = np.exp(z); p /= p.sum()
        p = np.maximum(p, self.floor)            # floor 防止饿死某个域
        return p / p.sum()
```

**bandit** 类 mixer(如 `dump_ucb`)的 `act` 还会通过 `ctx` 收到 `counts=`(每域访问次数)以加
探索项。

## 第 2 步:带 domain 列的数据

保留真实 `data_source`(verl 的 reward 路由依赖它);域标签放单独列。mix trainer 读
`config.dataflex.domain_key`(默认 `domain`)并给每个 prompt 打标签,供 replay buffer 按域分桶。

## 第 3 步:使用

```yaml
trainer:
  v1:
    trainer_mode: dataflex_mix_sync
    sampler:
      custom_sampler: {path: pkg://dataflex_verl.replay_buffer, name: DataFlexMixReplayBuffer}
dataflex:
  mechanism: mix
  domains: [math, code, logic]
  scorer:   {name: reward_difficulty}
  actuator: {name: reward_gap, params: {temperature: 1.0, floor: 0.05}}
  warmup_step: 5
  update_step: 5
```

## 端到端流程

1. `DataFlexMixSyncTrainer._add_batch_to_generate` 给每个 prompt 打域标签。
2. reward/advantage 之后,`DomainStatsTracker` 在滑动窗口上累积每域平均信号。
3. warmup 之后,每 `update_step` 步,`mixer.act(stats)` 产出新配比,写入共享状态。
4. `DataFlexMixReplayBuffer.sample` 把可采样 prompt 按域分桶,按配比选择(最大余数取整,并有
   oldest-first 兜底,保证不返回空 batch)。

由于 Mix 只周期性更新且需要跨步统计,它是唯一保持"训一段 → 调整 → 继续"节奏的机制。
