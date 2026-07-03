---
title: Add a Reweighter
icon: solar:add-circle-outline
createTime: 2026/07/03 10:00:00
permalink: /en/guide/reweighter/tutorial/
---

# Add a Reweighter to DataFlex-RL

This walks through adding a custom Reweighter. Because scoring is a shared layer, you
usually add **a Scorer (if the signal is new)** and **a Reweighter (the weighting
rule)** — and one Scorer can feed Selectors and Mixers too.

## Step 1: (optional) Add a Scorer

If your signal isn't already available, add a Scorer in
`src/dataflex_verl/scorers.py`. It declares what batch fields it reads and returns a
score:

```python
from .core.registry import register_scorer
from .core.scorer import Scorer

@register_scorer("advantage_magnitude")
class AdvantageMagnitudeScorer(Scorer):
    requires = ["advantages", "response_mask"]   # fields read from the batch
    timing = "post_advantage"
    granularity = "prompt"                        # per-sample score (bs,)
    needs_groups = False                          # works for any estimator

    def score(self, batch, step_id, **ctx):
        adv = batch.batch["advantages"]           # (bs, L)
        mask = batch.batch["response_mask"].to(adv.dtype)
        a = adv.abs() * mask
        return a.sum(-1) / mask.sum(-1).clamp(min=1.0)   # (bs,)
```

**Key points:**
- `@register_scorer("name")` registers it under a name used in config.
- `requires` drives an automatic compatibility check at mount time; `needs_groups=True`
  makes the framework reject the scorer on non-group estimators (PPO+GAE).
- `granularity="token"` tells the trainer to expect a `(bs, L)` score and skip the
  per-sample→per-token broadcast (used by token-level reweighters).

## Step 2: Add the Reweighter

In `src/dataflex_verl/reweighters.py`, subclass `Reweighter` and implement `act`,
which turns the score into weights. Mean-normalize to preserve the effective LR.

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
        w = torch.softmax(s / self.temperature, dim=0) * s.numel()  # mean ~1
        return _normalize_mean_one(w)
```

For a **token-level** reweighter, pair it with a `granularity="token"` scorer and
return a `(bs, L)` weight matrix directly (see `advantage_reweight`).

## Step 3: Use it in config

```yaml
trainer: {v1: {trainer_mode: dataflex_sync}}
dataflex:
  mechanism: reweight
  scorer:   {name: advantage_magnitude}
  actuator: {name: softmax, params: {temperature: 1.0}}
```

## How it reaches the loss

The `dataflex_sync` trainer, in `_compute_advantage`, reads the scorer's required
fields back from the TransferQueue, runs `scorer.score → reweighter.act`, broadcasts
per-sample weights to per-token (or uses the token matrix directly), and writes them
to the `rollout_is_weights` field. verl's vanilla policy loss then multiplies them into
`pg_losses` — no custom loss. Always mean-normalize so the step size stays comparable
to the unweighted baseline.
