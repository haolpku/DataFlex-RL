---
title: Add a Mixer
icon: solar:add-circle-outline
createTime: 2026/07/03 10:00:00
permalink: /en/guide/mixer/tutorial/
---

# Add a Mixer to DataFlex-RL

A Mixer consumes **per-domain statistics** and returns sampling **proportions**. Unlike
Reweighter/Selector, it does not act on the current batch — it steers future sampling.

## Step 1: Add the Mixer

`act` receives a `{domain: score}` dict (from the trainer's sliding-window tracker) and
returns a proportion vector aligned to `self.domains`.

```python
import numpy as np
from .core.actuator import Mixer
from .core.registry import register_mixer

@register_mixer("reward_gap")
class RewardGapMixer(Mixer):
    """proportions ∝ softmax((max_reward − domain_reward) / T): favor lagging domains."""
    def __init__(self, domains, temperature=1.0, floor=0.05, **kw):
        super().__init__(**kw)
        self.domains = list(domains); self.temperature = temperature; self.floor = floor

    def act(self, scores, batch, **ctx) -> np.ndarray:
        means = np.array([scores.get(d, 0.0) for d in self.domains])
        z = (means.max() - means) / max(self.temperature, 1e-6)
        z -= z.max()
        p = np.exp(z); p /= p.sum()
        p = np.maximum(p, self.floor)            # floor prevents starving a domain
        return p / p.sum()
```

For a **bandit** mixer (e.g. `dump_ucb`), `act` also takes `counts=` (per-domain visit
counts) via `ctx` to add an exploration bonus.

## Step 2: Data with a domain column

Keep the real `data_source` (verl's reward routing depends on it); put the domain label
in a separate column. The mix trainer reads `config.dataflex.domain_key` (default
`domain`) and tags each prompt so the replay buffer can bucket by domain.

## Step 3: Use it

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

## How it works end-to-end

1. `DataFlexMixSyncTrainer._add_batch_to_generate` tags each prompt with its domain.
2. After reward/advantage, a `DomainStatsTracker` accumulates each domain's mean signal
   over a sliding window.
3. After warmup, every `update_step` steps, `mixer.act(stats)` produces new proportions,
   written to shared state.
4. `DataFlexMixReplayBuffer.sample` buckets sampleable prompts by domain and selects to
   match those proportions (largest-remainder rounding, with an oldest-first top-up so
   it never returns an empty batch).

Because Mix updates only periodically and needs cross-step stats, it is the one
mechanism that keeps the "train a while → adjust → continue" rhythm.
