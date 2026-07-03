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


@register_mixer("dump_ucb")
class DumpUCBMixer(Mixer):
    """UCB bandit over per-domain mean |advantage| (DUMP, arXiv:2504.09710).

    Treats each domain as an arm whose value is its learnability = E[|advantage|]
    over the window. UCB adds an exploration bonus for under-sampled domains, then
    softmax over UCB scores gives proportions. Unlike reward_gap (exploit-only), this
    explores; and |advantage| self-regulates (peaks at mid-difficulty), avoiding the
    over-investment in unsolvable domains that a raw reward gap can cause.

    ``scores`` is a dict {domain -> mean |advantage|}; visit counts come from ``ctx``
    (the trainer passes cumulative per-domain sample counts).
    """

    def __init__(self, domains, temperature: float = 1.0, c: float = 1.0, floor: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.domains = list(domains)
        self.temperature = temperature
        self.c = c            # exploration coefficient
        self.floor = floor

    def act(self, scores, batch, *, counts=None, **ctx) -> np.ndarray:
        counts = counts or {}
        vals = np.array([scores.get(d, 0.0) for d in self.domains], dtype=np.float64)
        n_d = np.array([counts.get(d, 0) for d in self.domains], dtype=np.float64)
        N = n_d.sum()
        bonus = self.c * np.sqrt(2.0 * np.log(N + 1.0) / (n_d + 1.0))
        ucb = vals + bonus
        z = ucb / max(self.temperature, 1e-6)
        z -= z.max()
        p = np.exp(z)
        p = p / p.sum()
        p = np.maximum(p, self.floor)
        return p / p.sum()
