"""Reweighter actuators: score -> per-sample weights.

The trainer broadcasts these per-sample weights to per-token and writes them into
the ``rollout_is_weights`` field, which verl's vanilla policy loss multiplies into
pg_losses before aggregation (see verl/workers/utils/losses.py). No custom policy
loss is needed — we reuse verl's existing per-token weight hook.
"""

import torch

from .core.actuator import Reweighter
from .core.registry import register_reweighter


def _normalize_mean_one(w: torch.Tensor) -> torch.Tensor:
    """Rescale weights so their mean is 1.0 — keeps the effective LR/step size
    comparable to the unweighted baseline (only the *relative* emphasis changes)."""
    m = w.mean().clamp(min=1e-8)
    return w / m


@register_reweighter("softmax")
class SoftmaxReweighter(Reweighter):
    """w_i = softmax(score_i / T), renormalized to mean 1. Emphasizes high-score samples."""

    def __init__(self, temperature: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def act(self, scores: torch.Tensor, batch, **ctx) -> torch.Tensor:
        s = scores.float().flatten()
        w = torch.softmax(s / self.temperature, dim=0) * s.numel()  # mean ~1
        return _normalize_mean_one(w)


@register_reweighter("difficulty_band")
class DifficultyBandReweighter(Reweighter):
    """Up-weight mid-difficulty samples, down-weight extremes.

    Given a per-sequence score (e.g. outcome reward), samples whose score falls in
    [low_q, high_q] quantiles get ``focus_weight``x emphasis; the rest get 1x.
    Renormalized to mean 1.
    """

    def __init__(self, low_q: float = 0.25, high_q: float = 0.75, focus_weight: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.low_q = low_q
        self.high_q = high_q
        self.focus_weight = focus_weight

    def act(self, scores: torch.Tensor, batch, **ctx) -> torch.Tensor:
        s = scores.float().flatten()
        if s.numel() <= 1:
            return torch.ones_like(s)
        q1 = torch.quantile(s, self.low_q)
        q2 = torch.quantile(s, self.high_q)
        w = torch.ones_like(s)
        band = (s >= q1) & (s <= q2)
        w[band] = self.focus_weight
        return _normalize_mean_one(w)
