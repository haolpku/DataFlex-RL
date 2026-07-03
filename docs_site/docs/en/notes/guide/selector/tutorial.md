---
title: Add a Selector
icon: solar:add-circle-outline
createTime: 2026/07/03 10:00:00
permalink: /en/guide/selector/tutorial/
---

# Add a Selector to DataFlex-RL

A Selector turns a score into a list of kept indices. It reuses the same Scorer layer
as Reweighters and Mixers.

## Step 1: (optional) Add a group Scorer

Group-based signals (e.g. per-group solve rate) set `needs_groups=True`, which the
framework validates against the active estimator at mount time.

```python
import numpy as np, torch
from .core.registry import register_scorer
from .core.scorer import Scorer

@register_scorer("group_solve_rate")
class GroupSolveRateScorer(Scorer):
    requires = ["rm_scores", "response_mask", "uid"]
    timing = "post_reward"
    granularity = "prompt"
    needs_groups = True                    # rejected on PPO+GAE at startup

    def __init__(self, success_threshold: float = 0.5, **kw):
        super().__init__(**kw); self.success_threshold = success_threshold

    def score(self, batch, step_id, **ctx):
        scores = batch.batch["rm_scores"]
        mask = batch.batch["response_mask"]
        per_seq = (scores * mask).sum(-1)
        success = (per_seq > self.success_threshold).float()
        uid = np.asarray(batch.non_tensor_batch["uid"])
        out = torch.zeros_like(per_seq)
        for g in np.unique(uid):            # each sample gets its group's solve rate
            idx = np.where(uid == g)[0]
            out[idx] = success[idx].mean()
        return out
```

## Step 2: Add the Selector

Return the indices to keep. Dropped samples get zero loss weight.

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

## Step 3: Use it

```yaml
dataflex:
  mechanism: select
  scorer:   {name: group_solve_rate, params: {success_threshold: 0.5}}
  actuator: {name: threshold_band, params: {low: 0.2, high: 0.8}}
```

## How it reaches the loss

In `_compute_advantage`, the trainer converts the kept-index list into a 0/1 per-sample
mask, broadcasts it to per-token, and writes `rollout_is_weights`. Dropped samples then
contribute zero to `pg_losses`. Reweight and Select share this hook because both reduce
to per-token weights.
