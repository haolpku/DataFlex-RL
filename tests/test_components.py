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


# ---------------- new: OPD distillation signal (distill_kl / distill_gap) ----------------

def _distill_batch(student_logp, teacher_logp, mask=None, teacher_topk=False):
    """Build a fake batch with student old_log_probs + teacher_logprobs.

    student_logp / teacher_logp: (bs, L) python lists or tensors.
    teacher_topk: if True, give teacher_logprobs a trailing (K) axis (sampled token
    at column 0) to exercise the 3-D squeeze path.
    """
    s = torch.as_tensor(student_logp, dtype=torch.float32)
    t = torch.as_tensor(teacher_logp, dtype=torch.float32)
    bs, L = s.shape
    if mask is None:
        mask = torch.ones(bs, L)
    tl = t.unsqueeze(-1) if teacher_topk else t          # (bs,L,1) or (bs,L)
    if teacher_topk:  # pad a 2nd (non-sampled) top-k column that must be ignored
        tl = torch.cat([t.unsqueeze(-1), torch.full((bs, L, 1), -99.0)], dim=-1)
    return FakeBatch({"old_log_probs": s, "teacher_logprobs": tl, "response_mask": mask})


def test_distill_kl_signed_abs_pos():
    # student more confident than teacher at t0 (k>0), less at t1 (k<0)
    student = [[-0.5, -2.0]]
    teacher = [[-1.5, -0.5]]                              # k = [+1.0, -1.5]
    for mode, expect in [("signed", [[1.0, -1.5]]),
                         ("abs",    [[1.0, 1.5]]),
                         ("pos",    [[1.0, 0.0]])]:
        s = REGISTRY.build("scorer", "distill_kl", runtime={}, cfg={"mode": mode})
        assert s.granularity == "token"
        out = s.score(_distill_batch(student, teacher), 0)
        assert out.shape == (1, 2)
        assert torch.allclose(out, torch.tensor(expect), atol=1e-6), (mode, out)


def test_distill_kl_teacher_topk_axis():
    # 3-D teacher_logprobs (sampled token at col 0) must give same result as 2-D
    student, teacher = [[-0.5, -2.0]], [[-1.5, -0.5]]
    s = REGISTRY.build("scorer", "distill_kl", runtime={}, cfg={"mode": "signed"})
    out2d = s.score(_distill_batch(student, teacher, teacher_topk=False), 0)
    out3d = s.score(_distill_batch(student, teacher, teacher_topk=True), 0)
    assert torch.allclose(out2d, out3d, atol=1e-6)


def test_distill_gap_seq_aggregate_and_mask():
    # two seqs; seq0 disagreement bigger. mask off the last token of seq1.
    student = [[-0.5, -2.0], [-1.0, -1.0]]
    teacher = [[-1.5, -0.5], [-1.2, 10.0]]               # |k| = [[1.0,1.5],[0.2,11.0]]
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])        # drop seq1 t1
    s = REGISTRY.build("scorer", "distill_gap", runtime={}, cfg={"mode": "abs"})
    assert s.granularity == "prompt"
    out = s.score(_distill_batch(student, teacher, mask=mask), 0)
    assert out.shape == (2,)
    # seq0 mean |k| = (1.0+1.5)/2 = 1.25 ; seq1 = 0.2/1 = 0.2 (t1 masked out)
    assert torch.allclose(out, torch.tensor([1.25, 0.2]), atol=1e-6)


def test_distill_missing_teacher_field_errors_clearly():
    s = REGISTRY.build("scorer", "distill_kl", runtime={}, cfg={})
    b = FakeBatch({"old_log_probs": torch.zeros(1, 2), "response_mask": torch.ones(1, 2)})
    import pytest
    with pytest.raises(KeyError, match="distillation.enabled"):
        s.score(b, 0)


def test_distill_gap_feeds_selector_and_reweighter():
    # end-to-end wiring: distill_gap -> selector keeps high-gap; distill_kl -> reweighter
    student = [[-0.5, -2.0], [-1.0, -1.0], [-0.9, -0.9]]
    teacher = [[-2.5, -3.0], [-1.05, -1.0], [-0.92, -0.9]]  # seq0 huge gap, seq2 tiny
    b = _distill_batch(student, teacher)
    gap = REGISTRY.build("scorer", "distill_gap", runtime={}, cfg={"mode": "abs"})
    sel = REGISTRY.build("selector", "topk_fraction", runtime={}, cfg={"fraction": 0.34})
    keep = sel.act(gap.score(b, 0), b)
    assert keep == [0]                                    # highest-gap seq survives
    kl = REGISTRY.build("scorer", "distill_kl", runtime={}, cfg={"mode": "abs"})
    rw = REGISTRY.build("reweighter", "advantage_reweight", runtime={}, cfg={"alpha": 0.5})
    # distill_kl is token-granularity; reweighter consumes (bs,L) and returns (bs,L)
    w = rw.act(kl.score(b, 0).clamp(max=1.0), b)          # clamp: reuse pi-in-(0,1] path
    assert w.shape == (3, 2)


# ---------------- new: OPD build-time compatibility validation ----------------

def _mk_distill_dataflex(mechanism, scorer="distill_gap"):
    """A config.dataflex block using a teacher scorer + a mechanism-appropriate actuator."""
    act = {"reweight": {"name": "advantage_reweight"},
           "select":   {"name": "topk_fraction", "params": {"fraction": 0.5}},
           "mix":      {"name": "reward_gap"}}[mechanism]
    sc = "distill_kl" if mechanism == "reweight" else scorer
    return {"mechanism": mechanism, "scorer": {"name": sc}, "actuator": act}


def test_opd_teacher_scorer_requires_distillation_enabled():
    import pytest
    # distillation off -> teacher scorer rejected at build time
    with pytest.raises(ValueError, match="distillation.enabled"):
        build_from_config(_mk_distill_dataflex("reweight"), adv_estimator="grpo",
                          distillation={"enabled": False})


def test_opd_reweight_select_reject_gkd():
    import pytest
    gkd = {"enabled": True, "distillation_loss": {"use_policy_gradient": False}}
    for mech in ("reweight", "select"):
        with pytest.raises(ValueError, match="PG OPD"):
            build_from_config(_mk_distill_dataflex(mech),
                              adv_estimator="grpo", distillation=gkd)


def test_opd_reweight_select_ok_under_pg():
    pg = {"enabled": True, "distillation_loss": {"use_policy_gradient": True}}
    for mech in ("reweight", "select"):
        scorer, act, meta = build_from_config(_mk_distill_dataflex(mech),
                                              adv_estimator="grpo", distillation=pg)
        assert meta["mechanism"] == mech          # builds without raising


def test_opd_mix_allowed_under_any_mode():
    # mix does not depend on the loss path, so GKD is fine
    gkd = {"enabled": True, "distillation_loss": {"use_policy_gradient": False}}
    scorer, mixer, meta = build_from_config(
        {"mechanism": "mix", "scorer": {"name": "distill_gap"}, "actuator": {"name": "reward_gap"}},
        adv_estimator="grpo", runtime={"domains": ["a", "b"]}, distillation=gkd)
    assert meta["mechanism"] == "mix"


def test_non_teacher_scorer_unaffected_by_opd_check():
    # a normal reward scorer builds fine regardless of distillation config
    scorer, act, meta = build_from_config(
        {"mechanism": "reweight", "scorer": {"name": "advantage_magnitude"},
         "actuator": {"name": "per_advantage", "params": {"alpha": 0.5}}},
        adv_estimator="grpo", distillation=None)
    assert meta["mechanism"] == "reweight"


# ---------------- new: M4 divergence-driven mix (per-domain distill_gap) ----------------

def _agg_token_to_seq(scorer, batch):
    """Replicate DataFlexMixSyncTrainer._per_seq_signal for CPU testing (no verl)."""
    scores = scorer.score(batch, 0)
    if getattr(scorer, "granularity", "prompt") == "token":
        mask = batch.batch["response_mask"].to(scores.dtype)
        denom = mask.sum(dim=-1).clamp(min=1.0)
        return (scores * mask).sum(dim=-1) / denom
    return scores.flatten()


def test_mix_signal_distill_gap_per_domain_drives_proportions():
    # 2 domains: 'math' has large teacher-student divergence, 'code' small.
    # distill_gap -> per-seq gap -> per-domain mean -> reward_gap mixer.
    from dataflex_verl.mixers import DomainStatsTracker
    student = [[-0.5, -2.0], [-0.6, -1.8],   # math seqs (big gap)
               [-0.9, -0.9], [-0.95, -0.9]]  # code seqs (tiny gap)
    teacher = [[-2.5, -4.0], [-2.4, -3.6],
               [-0.92, -0.9], [-0.96, -0.9]]
    b = _distill_batch(student, teacher)
    gap = REGISTRY.build("scorer", "distill_gap", runtime={}, cfg={"mode": "abs"})
    per_seq = _agg_token_to_seq(gap, b)
    assert per_seq.shape == (4,)
    assert per_seq[:2].mean() > per_seq[2:].mean()        # math gap > code gap

    domains = ["math", "code"]
    tr = DomainStatsTracker(window=50)
    for d, v in zip(["math", "math", "code", "code"], per_seq.tolist()):
        tr.update(d, v)
    stats = {d: tr.mean(d) for d in domains}
    mx = REGISTRY.build("mixer", "reward_gap", runtime={"domains": domains},
                        cfg={"temperature": 1.0, "floor": 0.0})
    props = mx.act(stats, b)
    assert abs(props.sum() - 1.0) < 1e-9
    # reward_gap favors the LAGGING domain; here the "signal" is divergence, so the
    # higher-divergence domain (math) is treated as lagging -> gets MORE. Just assert
    # proportions respond to the per-domain signal (not uniform).
    assert abs(props[0] - props[1]) > 1e-3
