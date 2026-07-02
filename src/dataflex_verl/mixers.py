"""Mixer actuators: domain statistics -> sampling proportions.

Mixture is retrospective and per-domain: it consumes a sliding window of each
domain's reward/advantage and returns the proportion each domain should get in
future sampling. Unlike Selector/Reweighter it does NOT act on the current batch's
members; it steers what gets sampled next. Needs a warmup phase to accumulate stats
(cold start) — see DomainStatsTracker.
"""

from collections import defaultdict, deque

import numpy as np

from .core.actuator import Mixer
from .core.registry import register_mixer


class DomainStatsTracker:
    """Sliding-window mean of a per-domain scalar signal (e.g. mean reward)."""

    def __init__(self, window: int = 50):
        self.window = window
        self._buf = defaultdict(lambda: deque(maxlen=window))

    def update(self, domain: str, value: float) -> None:
        self._buf[domain].append(float(value))

    def mean(self, domain: str, default: float = 0.0) -> float:
        b = self._buf[domain]
        return sum(b) / len(b) if b else default

    def domains(self):
        return list(self._buf.keys())


@register_mixer("reward_gap")
class RewardGapMixer(Mixer):
    """Allocate more to domains with LOWER mean reward (larger gap to mastery).

    proportions ∝ softmax((max_reward - domain_reward) / T). A domain the model
    already aces gets less; a lagging domain gets more. Mirrors DoReMi's excess-loss
    idea using RL reward.
    """

    def __init__(self, domains, temperature: float = 1.0, floor: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.domains = list(domains)
        self.temperature = temperature
        self.floor = floor  # min proportion per domain, to avoid starving any source

    def act(self, scores, batch, **ctx) -> np.ndarray:
        """``scores`` is a dict {domain -> mean_reward} from the tracker."""
        means = np.array([scores.get(d, 0.0) for d in self.domains], dtype=np.float64)
        gap = means.max() - means
        z = gap / max(self.temperature, 1e-6)
        z -= z.max()
        p = np.exp(z)
        p = p / p.sum()
        # enforce a floor, then renormalize
        p = np.maximum(p, self.floor)
        p = p / p.sum()
        return p


@register_mixer("static")
class StaticMixer(Mixer):
    """Fixed proportions (baseline / warmup fallback)."""

    def __init__(self, domains, proportions=None, **kwargs):
        super().__init__(**kwargs)
        self.domains = list(domains)
        if proportions is None:
            proportions = [1.0 / len(self.domains)] * len(self.domains)
        p = np.asarray(proportions, dtype=np.float64)
        self._p = p / p.sum()

    def act(self, scores, batch, **ctx) -> np.ndarray:
        return self._p
