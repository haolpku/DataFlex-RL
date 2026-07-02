"""Offline tests for scorers / actuators / build — no verl, no GPU.

Uses a tiny fake batch that mimics verl's DataProto interface (.batch dict of
tensors + .non_tensor_batch dict), which is all our scorers touch.
"""

import numpy as np
import torch

from dataflex_verl.build import build_from_config
from dataflex_verl.core.registry import REGISTRY
import dataflex_verl.scorers  # noqa: F401  (register)
import dataflex_verl.reweighters  # noqa: F401
import dataflex_verl.selectors  # noqa: F401
import dataflex_verl.mixers  # noqa: F401


class FakeBatch:
    def __init__(self, batch, non_tensor_batch=None):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch or {}


def make_batch(rewards, advs=None, uids=None):
    """rewards: list of per-seq outcome rewards. Spread onto a (bs, L) token grid."""
    bs = len(rewards)
    L = 4
    mask = torch.ones(bs, L)
    tls = torch.zeros(bs, L)
    tls[:, -1] = torch.tensor(rewards, dtype=torch.float32)  # outcome at last token
    b = {"token_level_scores": tls, "response_mask": mask}
    if advs is not None:
        a = torch.zeros(bs, L)
        a[:, -1] = torch.tensor(advs, dtype=torch.float32)
        b["advantages"] = a
    nt = {}
    if uids is not None:
        nt["uid"] = np.array(uids, dtype=object)
    return FakeBatch(b, nt)


# ---------------- scorers ----------------

def test_reward_difficulty_scorer():
    s = REGISTRY.build("scorer", "reward_difficulty", runtime={}, cfg={})
    batch = make_batch([1.0, 0.0, 0.5])
    out = s.score(batch, 0)
    assert torch.allclose(out, torch.tensor([1.0, 0.0, 0.5]))


def test_advantage_magnitude_scorer():
    s = REGISTRY.build("scorer", "advantage_magnitude", runtime={}, cfg={"agg": "sum"})
    batch = make_batch([0, 0, 0], advs=[-2.0, 1.0, 0.0])
    out = s.score(batch, 0)
    assert torch.allclose(out, torch.tensor([2.0, 1.0, 0.0]))


def test_group_solve_rate_scorer():
    s = REGISTRY.build("scorer", "group_solve_rate", runtime={}, cfg={"success_threshold": 0.5})
    # group A: [1,0] -> rate .5 ; group B: [1,1] -> rate 1.0 ; group C: [0,0] -> 0
    batch = make_batch([1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                       uids=["A", "A", "B", "B", "C", "C"])
    out = s.score(batch, 0)
    assert torch.allclose(out, torch.tensor([0.5, 0.5, 1.0, 1.0, 0.0, 0.0]))


# ---------------- reweighters ----------------

def test_softmax_reweighter_mean_one():
    rw = REGISTRY.build("reweighter", "softmax", runtime={}, cfg={"temperature": 1.0})
    scores = torch.tensor([1.0, 2.0, 3.0])
    w = rw.act(scores, None)
    assert abs(float(w.mean()) - 1.0) < 1e-5
    # higher score -> higher weight
    assert w[2] > w[1] > w[0]


def test_difficulty_band_reweighter():
    rw = REGISTRY.build("reweighter", "difficulty_band", runtime={},
                        cfg={"low_q": 0.25, "high_q": 0.75, "focus_weight": 3.0})
    scores = torch.tensor([0.0, 0.5, 0.5, 1.0])  # middle band emphasized
    w = rw.act(scores, None)
    assert abs(float(w.mean()) - 1.0) < 1e-5
    assert w[1] > w[0] and w[2] > w[3]


# ---------------- selectors ----------------

def test_threshold_band_selector_dapo():
    sel = REGISTRY.build("selector", "threshold_band", runtime={}, cfg={"low": 0.0, "high": 1.0})
    # group solve rates: keep only the strictly-between-0-and-1 group
    scores = torch.tensor([0.0, 0.0, 0.5, 0.5, 1.0, 1.0])
    keep = sel.act(scores, None)
    assert keep == [2, 3]


def test_topk_fraction_selector():
    sel = REGISTRY.build("selector", "topk_fraction", runtime={}, cfg={"fraction": 0.5, "largest": True})
    scores = torch.tensor([0.1, 0.9, 0.3, 0.7])
    keep = sel.act(scores, None)
    assert keep == [1, 3]


# ---------------- mixers ----------------

def test_reward_gap_mixer_favors_low_reward():
    mx = REGISTRY.build("mixer", "reward_gap", runtime={"domains": ["math", "code"]},
                        cfg={"temperature": 0.5, "floor": 0.0})
    # code has lower reward -> should get MORE proportion
    props = mx.act({"math": 0.9, "code": 0.1}, None)
    assert props.sum() == 1.0 or abs(props.sum() - 1.0) < 1e-9
    domains = ["math", "code"]
    assert props[domains.index("code")] > props[domains.index("math")]


def test_static_mixer():
    mx = REGISTRY.build("mixer", "static", runtime={"domains": ["a", "b", "c"]}, cfg={})
    props = mx.act({}, None)
    assert abs(props.sum() - 1.0) < 1e-9
    assert np.allclose(props, [1 / 3, 1 / 3, 1 / 3])


# ---------------- build_from_config ----------------

def test_build_reweight_config():
    cfg = {
        "mechanism": "reweight",
        "scorer": {"name": "advantage_magnitude", "params": {"agg": "mean"}},
        "actuator": {"name": "softmax", "params": {"temperature": 2.0}},
        "warmup_step": 5,
    }
    scorer, actuator, meta = build_from_config(cfg, adv_estimator="grpo")
    assert scorer.requires == ["advantages", "response_mask"]
    assert meta["mechanism"] == "reweight"
    assert meta["warmup_step"] == 5


def test_build_rejects_group_scorer_on_gae():
    cfg = {
        "mechanism": "select",
        "scorer": {"name": "group_solve_rate", "params": {}},
        "actuator": {"name": "threshold_band", "params": {}},
    }
    import pytest
    with pytest.raises(ValueError, match="group"):
        build_from_config(cfg, adv_estimator="gae")


def test_build_mix_needs_domains_runtime():
    cfg = {
        "mechanism": "mix",
        "scorer": {"name": "reward_difficulty", "params": {}},
        "actuator": {"name": "reward_gap", "params": {"temperature": 1.0}},
    }
    scorer, mixer, meta = build_from_config(cfg, adv_estimator="grpo", runtime={"domains": ["a", "b"]})
    assert meta["mechanism"] == "mix"
    props = mixer.act({"a": 0.1, "b": 0.9}, None)
    assert abs(props.sum() - 1.0) < 1e-9
