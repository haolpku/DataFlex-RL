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


@register_reweighter("advantage_reweight")
class AdvantageReweighter(Reweighter):
    """Token-level low-probability damping (AR, arXiv:2505.12929).

    w_t = alpha * pi_theta(t) + (1 - alpha), where pi_theta(t) is the per-token
    probability from the ``token_prob`` scorer. Low-prob tokens (small pi) get a
    weight near (1-alpha), high-prob tokens near 1 — damping the outsized gradients
    low-prob tokens would otherwise contribute. Mean-normalized over valid tokens.

    Expects a (bs, L) score (per-token prob) and returns a (bs, L) weight matrix;
    paired with a token-granularity scorer so the trainer skips the per-sample
    broadcast.
    """

    def __init__(self, alpha: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        assert 0.0 <= alpha <= 1.0
        self.alpha = alpha

    def act(self, scores: torch.Tensor, batch, **ctx) -> torch.Tensor:
        pi = scores.float()                                   # (bs, L) in (0,1]
        w = self.alpha * pi + (1.0 - self.alpha)              # (bs, L)
        # mean-normalize over valid tokens to preserve effective LR
        mask = batch.batch["response_mask"].to(w.dtype)
        denom = (w * mask).sum().clamp(min=1e-8)
        cnt = mask.sum().clamp(min=1.0)
        return w * (cnt / denom)
