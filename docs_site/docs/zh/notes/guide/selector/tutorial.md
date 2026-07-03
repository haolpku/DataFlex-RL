---
title: 添加 Selector
icon: solar:add-circle-outline
createTime: 2026/07/03 10:00:00
permalink: /zh/guide/selector/tutorial/
---

# 向 DataFlex-RL 添加 Selector

Selector 把分数转成保留索引列表,复用与 Reweighter/Mixer 相同的 Scorer 层。

## 第 1 步:(可选)添加分组 Scorer

分组信号(如组解出率)设 `needs_groups=True`,框架在挂载时对当前估计器做校验。

```python
import numpy as np, torch
from .core.registry import register_scorer
from .core.scorer import Scorer

@register_scorer("group_solve_rate")
class GroupSolveRateScorer(Scorer):
    requires = ["rm_scores", "response_mask", "uid"]
    timing = "post_reward"
    granularity = "prompt"
    needs_groups = True                    # 在 PPO+GAE 上启动时被拒

    def __init__(self, success_threshold: float = 0.5, **kw):
        super().__init__(**kw); self.success_threshold = success_threshold

    def score(self, batch, step_id, **ctx):
        scores = batch.batch["rm_scores"]
        mask = batch.batch["response_mask"]
        per_seq = (scores * mask).sum(-1)
        success = (per_seq > self.success_threshold).float()
        uid = np.asarray(batch.non_tensor_batch["uid"])
        out = torch.zeros_like(per_seq)
        for g in np.unique(uid):            # 每个样本取其所在组的解出率
            idx = np.where(uid == g)[0]
            out[idx] = success[idx].mean()
        return out
```

## 第 2 步:添加 Selector

返回要保留的索引。被丢弃样本的 loss 权重为零。

```python
import torch
from .core.actuator import Selector
from .core.registry import register_selector

@register_selector("threshold_band")
class ThresholdBandSelector(Selector):
    def __init__(self, low: float = 0.0, high: float = 1.0, **kw):
        super().__init__(**kw); self.low = low; self.high = high

    def act(self, scores, batch, **ctx) -> list[int]:
        s = scores.float().flatten()
        keep = (s > self.low) & (s < self.high)
        return torch.nonzero(keep, as_tuple=False).flatten().tolist()
```

## 第 3 步:使用

```yaml
dataflex:
  mechanism: select
  scorer:   {name: group_solve_rate, params: {success_threshold: 0.5}}
  actuator: {name: threshold_band, params: {low: 0.2, high: 0.8}}
```

## 如何抵达 loss

在 `_compute_advantage` 里,trainer 把保留索引列表转成 0/1 的 per-sample 掩码,广播到
per-token,写入 `rollout_is_weights`。被丢弃样本对 `pg_losses` 贡献为零。Reweight 和 Select
共享这个钩子,因为两者都归结为 per-token 权重。
