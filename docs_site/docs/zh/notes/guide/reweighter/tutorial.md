---
title: 添加 Reweighter
icon: solar:add-circle-outline
createTime: 2026/07/03 10:00:00
permalink: /zh/guide/reweighter/tutorial/
---

# 向 DataFlex-RL 添加 Reweighter

由于打分是共享层,通常你要加**一个 Scorer(如果信号是新的)**和**一个 Reweighter(加权规则)**
—— 而一个 Scorer 也能同时喂给 Selector 和 Mixer。

## 第 1 步:(可选)添加 Scorer

若信号还没有,在 `src/dataflex_verl/scorers.py` 里加一个 Scorer。它声明读哪些 batch 字段,
返回分数:

```python
from .core.registry import register_scorer
from .core.scorer import Scorer

@register_scorer("advantage_magnitude")
class AdvantageMagnitudeScorer(Scorer):
    requires = ["advantages", "response_mask"]   # 从 batch 读的字段
    timing = "post_advantage"
    granularity = "prompt"                        # per-sample 分数 (bs,)
    needs_groups = False                          # 任意估计器可用

    def score(self, batch, step_id, **ctx):
        adv = batch.batch["advantages"]           # (bs, L)
        mask = batch.batch["response_mask"].to(adv.dtype)
        a = adv.abs() * mask
        return a.sum(-1) / mask.sum(-1).clamp(min=1.0)   # (bs,)
```

**要点:**
- `@register_scorer("name")` 用一个名字注册,配置里引用。
- `requires` 驱动挂载时的自动兼容性检查;`needs_groups=True` 会让框架在非分组估计器
  (PPO+GAE)上拒绝该 scorer。
- `granularity="token"` 告诉 trainer 期望 `(bs, L)` 分数并跳过 per-sample→per-token 广播
  (token 级 reweighter 用)。

## 第 2 步:添加 Reweighter

在 `src/dataflex_verl/reweighters.py` 里继承 `Reweighter` 实现 `act`,把分数转成权重。
归一化到均值 1 以保持等效学习率。

```python
import torch
from .core.actuator import Reweighter
from .core.registry import register_reweighter

def _normalize_mean_one(w):
    return w / w.mean().clamp(min=1e-8)

@register_reweighter("softmax")
class SoftmaxReweighter(Reweighter):
    def __init__(self, temperature: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def act(self, scores, batch, **ctx):
        s = scores.float().flatten()
        w = torch.softmax(s / self.temperature, dim=0) * s.numel()  # 均值 ~1
        return _normalize_mean_one(w)
```

**token 级** reweighter:搭配 `granularity="token"` 的 scorer,直接返回 `(bs, L)` 权重矩阵
(见 `advantage_reweight`)。

## 第 3 步:在配置里用

```yaml
trainer: {v1: {trainer_mode: dataflex_sync}}
dataflex:
  mechanism: reweight
  scorer:   {name: advantage_magnitude}
  actuator: {name: softmax, params: {temperature: 1.0}}
```

## 如何抵达 loss

`dataflex_sync` trainer 在 `_compute_advantage` 里从 TransferQueue 读回 scorer 所需字段,
执行 `scorer.score → reweighter.act`,把 per-sample 权重广播到 per-token(或直接用 token 矩阵),
写入 `rollout_is_weights` 字段。verl 的 vanilla policy loss 随后把它乘进 `pg_losses` —— 无需
自定义 loss。务必归一化,使步长与无权重基线可比。
