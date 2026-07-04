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


# ---------------- new: token-level Advantage Reweighting ----------------

def test_token_prob_scorer_shape_and_range():
    s = REGISTRY.build("scorer", "token_prob", runtime={}, cfg={})
    assert s.granularity == "token"
    bs, L = 3, 4
    logp = torch.log(torch.full((bs, L), 0.5))          # pi = 0.5 everywhere
    b = FakeBatch({"old_log_probs": logp, "response_mask": torch.ones(bs, L)})
    out = s.score(b, 0)
    assert out.shape == (bs, L)
    assert torch.allclose(out, torch.full((bs, L), 0.5), atol=1e-6)


def test_advantage_reweighter_damps_low_prob_and_mean_one():
    rw = REGISTRY.build("reweighter", "advantage_reweight", runtime={}, cfg={"alpha": 0.5})
    # token probs: one low (0.1), one high (0.9)
    pi = torch.tensor([[0.1, 0.9]])
    mask = torch.ones(1, 2)
    b = FakeBatch({"response_mask": mask})
    w = rw.act(pi, b)
    assert w.shape == (1, 2)
    # low-prob token gets a smaller weight than high-prob token
    assert w[0, 0] < w[0, 1]
    # mean over valid tokens ~ 1
    assert abs(float((w * mask).sum() / mask.sum()) - 1.0) < 1e-5


# ---------------- new: DUMP-UCB mixer ----------------

def test_dump_ucb_explores_undersampled_domain():
    mx = REGISTRY.build("mixer", "dump_ucb", runtime={"domains": ["a", "b"]},
                        cfg={"temperature": 1.0, "c": 1.0, "floor": 0.0})
    # equal learnability, but domain b barely sampled -> b should get more (exploration)
    props = mx.act({"a": 0.5, "b": 0.5}, None, counts={"a": 100, "b": 1})
    assert abs(props.sum() - 1.0) < 1e-9
    assert props[1] > props[0]


def test_dump_ucb_exploits_high_advantage():
    mx = REGISTRY.build("mixer", "dump_ucb", runtime={"domains": ["a", "b"]},
                        cfg={"temperature": 0.5, "c": 0.1, "floor": 0.0})
    # same counts, a has higher learnability -> a gets more
    props = mx.act({"a": 1.0, "b": 0.1}, None, counts={"a": 50, "b": 50})
    assert props[0] > props[1]


# ---------------- new: GFPO / max_variance selectors ----------------

def _batch_with_uid_len(rewards, uids, lengths):
    import numpy as np
    bs = len(rewards); L = max(lengths)
    mask = torch.zeros(bs, L)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1.0
    tls = torch.zeros(bs, L); 
    for i, r in enumerate(rewards): tls[i, -1] = r
    b = FakeBatch({"token_level_scores": tls, "response_mask": mask},
                  {"uid": np.array(uids, dtype=object)})
    return b


def test_gfpo_efficiency_keeps_high_reward_per_len():
    sel = REGISTRY.build("selector", "gfpo", runtime={}, cfg={"k": 1, "metric": "efficiency"})
    # one group of 3: same reward=1, different lengths -> keep the shortest (best R/L)
    b = _batch_with_uid_len([1.0, 1.0, 1.0], ["g", "g", "g"], [10, 5, 20])
    scores = torch.tensor([1.0, 1.0, 1.0])
    keep = sel.act(scores, b)
    assert keep == [1]  # index 1 has length 5 -> highest reward/length


def test_gfpo_short_keeps_shortest():
    sel = REGISTRY.build("selector", "gfpo", runtime={}, cfg={"k": 2, "metric": "short"})
    b = _batch_with_uid_len([1, 1, 1, 1], ["g","g","g","g"], [8, 2, 6, 4])
    keep = sel.act(torch.tensor([1.,1.,1.,1.]), b)
    assert set(keep) == {1, 3}  # two shortest: len 2 and 4


def test_max_variance_binary_picks_extremes():
    sel = REGISTRY.build("selector", "max_variance", runtime={}, cfg={"keep_fraction": 0.5})
    # group of 4 binary rewards [1,1,0,0]; keep n=2 maximizing variance -> one 1 + one 0
    b = _batch_with_uid_len([1, 1, 0, 0], ["g","g","g","g"], [3,3,3,3])
    keep = sel.act(torch.tensor([1.,1.,0.,0.]), b)
    kept_r = sorted([1,1,0,0][i] for i in keep)
    assert kept_r == [0, 1]  # one low + one high


# ---------------- new: PER advantage reweighter ----------------

def test_per_advantage_weights():
    rw = REGISTRY.build("reweighter", "per_advantage", runtime={}, cfg={"alpha": 1.0})
    scores = torch.tensor([0.0, 1.0, 2.0, 3.0])   # |A|
    w = rw.act(scores, None)
    assert abs(float(w.mean()) - 1.0) < 1e-5
    assert w[3] > w[2] > w[1] > w[0]  # higher |A| -> higher weight


def test_per_advantage_alpha_zero_uniform():
    rw = REGISTRY.build("reweighter", "per_advantage", runtime={}, cfg={"alpha": 0.0})
    w = rw.act(torch.tensor([0.1, 5.0, 100.0]), None)
    assert torch.allclose(w, torch.ones(3), atol=1e-4)  # alpha=0 -> all ~1


# ---------------- new: TSCL mixer ----------------

def test_tscl_mixer_favors_steep_slope():
    mx = REGISTRY.build("mixer", "tscl", runtime={"domains": ["a", "b"]},
                        cfg={"temperature": 0.5, "floor": 0.0})
    # domain a improving fast (slope 0.5), b flat (0) -> a gets more
    props = mx.act({"a": 0.5, "b": 0.0}, None)
    assert abs(props.sum() - 1.0) < 1e-9
    assert props[0] > props[1]


def test_tscl_abs_slope_forgetting_also_sampled():
    mx = REGISTRY.build("mixer", "tscl", runtime={"domains": ["a", "b"]},
                        cfg={"temperature": 0.5, "floor": 0.0, "signed": False})
    # b is forgetting (slope -0.5); |slope| -> b sampled MORE than flat a
    props = mx.act({"a": 0.0, "b": -0.5}, None)
    assert props[1] > props[0]


def test_domain_tracker_slope():
    from dataflex_verl.mixers import DomainStatsTracker
    t = DomainStatsTracker(window=10)
    for v in [0.1, 0.2, 0.3, 0.4, 0.5]:   # rising
        t.update("d", v)
    assert t.slope("d") > 0.05
