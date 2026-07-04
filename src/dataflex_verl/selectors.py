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


import numpy as np


def _groups(batch, n):
    """Yield (indices) per uid group; falls back to one big group if no uid."""
    uid = batch.non_tensor_batch.get("uid") if hasattr(batch, "non_tensor_batch") else None
    if uid is None:
        yield list(range(n))
        return
    uid = np.asarray(uid)
    for g in np.unique(uid):
        yield np.where(uid == g)[0].tolist()


@register_selector("gfpo")
class GFPOSelector(Selector):
    """Group Filtered Policy Optimization (arXiv:2508.09726).

    Within each uid group keep the top-k responses by a filter metric; the rest are
    dropped (advantage 0). Two metrics:
      - metric="short"       : keep the k SHORTEST responses (conciseness).
      - metric="efficiency"  : keep the k highest reward/length (token efficiency).
    ``scores`` is the per-sample reward (from a reward scorer); response length is
    read from response_mask.
    """

    def __init__(self, k: int = 8, metric: str = "efficiency", **kwargs):
        super().__init__(**kwargs)
        assert metric in ("short", "efficiency")
        self.k = k
        self.metric = metric

    def act(self, scores: torch.Tensor, batch, **ctx) -> list[int]:
        R = scores.float().flatten()
        L = batch.batch["response_mask"].float().sum(-1).clamp(min=1.0)  # (bs,)
        if self.metric == "short":
            s = -L
        else:  # efficiency
            s = R / L
        keep = []
        n = R.numel()
        for idx in _groups(batch, n):
            if len(idx) <= self.k:
                keep.extend(idx)
                continue
            si = s[torch.as_tensor(idx, dtype=torch.long)]
            top = torch.topk(si, self.k, largest=True).indices.tolist()
            keep.extend([idx[j] for j in top])
        return sorted(keep)


@register_selector("max_variance")
class MaxVarianceSelector(Selector):
    """PODS max-variance down-sampling (arXiv:2504.13818).

    Within each uid group of size G, keep the size-n subset that MAXIMIZES reward
    variance. By the extreme-anchored theorem the optimum is `a` lowest + (n-a)
    highest rewards; we scan a=0..n with prefix sums. Binary rewards reduce to equal
    counts of high/low. ``scores`` is per-sample reward; ``keep_fraction`` sets n=round(f*G).
    """

    def __init__(self, keep_fraction: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        assert 0.0 < keep_fraction <= 1.0
        self.keep_fraction = keep_fraction

    def _best_subset(self, R_group):
        G = len(R_group)
        n = max(1, int(round(self.keep_fraction * G)))
        if n >= G:
            return list(range(G))
        order = sorted(range(G), key=lambda i: R_group[i])   # ascending by reward
        sr = [R_group[i] for i in order]
        # prefix sums of value and value^2
        ps = [0.0] * (G + 1)
        ps2 = [0.0] * (G + 1)
        for i in range(G):
            ps[i + 1] = ps[i] + sr[i]
            ps2[i + 1] = ps2[i] + sr[i] * sr[i]
        best_var, best_a = -1.0, 0
        for a in range(0, n + 1):
            b = n - a
            # a lowest: order[0:a] ; b highest: order[G-b:G]
            s = ps[a] + (ps[G] - ps[G - b] if b > 0 else 0.0)
            s2 = ps2[a] + (ps2[G] - ps2[G - b] if b > 0 else 0.0)
            var = s2 / n - (s / n) ** 2
            if var > best_var:
                best_var, best_a = var, a
        a = best_a
        b = n - a
        sel_local = order[:a] + (order[G - b:] if b > 0 else [])
        return sel_local

    def act(self, scores: torch.Tensor, batch, **ctx) -> list[int]:
        R = scores.float().flatten().tolist()
        n_total = len(R)
        keep = []
        for idx in _groups(batch, n_total):
            Rg = [R[i] for i in idx]
            sel_local = self._best_subset(Rg)
            keep.extend([idx[j] for j in sel_local])
        return sorted(keep)
