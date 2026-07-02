"""Selector actuators: score -> indices to keep.

In RL, dropping a sample here (before the actor update) removes its contribution
to the gradient. When done in the replay buffer *before rollout*, it also saves the
generation cost — the key RL-specific reason Selection != Reweighting-with-zero.
"""

import torch

from .core.actuator import Selector
from .core.registry import register_selector


@register_selector("threshold_band")
class ThresholdBandSelector(Selector):
    """Keep samples whose score is strictly inside (low, high).

    With a group_solve_rate scorer this reproduces DAPO dynamic sampling: drop
    all-solved (rate>=high) and all-failed (rate<=low) groups, since they carry
    zero advantage signal.
    """

    def __init__(self, low: float = 0.0, high: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.low = low
        self.high = high

    def act(self, scores: torch.Tensor, batch, **ctx) -> list[int]:
        s = scores.float().flatten()
        keep = (s > self.low) & (s < self.high)
        idx = torch.nonzero(keep, as_tuple=False).flatten().tolist()
        return idx


@register_selector("topk_fraction")
class TopKFractionSelector(Selector):
    """Keep the top ``fraction`` of samples by score (e.g. hardest / highest-|adv|)."""

    def __init__(self, fraction: float = 0.5, largest: bool = True, **kwargs):
        super().__init__(**kwargs)
        assert 0.0 < fraction <= 1.0
        self.fraction = fraction
        self.largest = largest

    def act(self, scores: torch.Tensor, batch, **ctx) -> list[int]:
        s = scores.float().flatten()
        k = max(1, int(round(self.fraction * s.numel())))
        idx = torch.topk(s, k, largest=self.largest).indices.tolist()
        return sorted(idx)
