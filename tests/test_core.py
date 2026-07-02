"""Unit tests for dataflex_verl.core: registry build semantics, scorer contracts,
and compatibility validation. Framework-agnostic — no torch/verl required."""

import pytest

from dataflex_verl.core import (
    REGISTRY,
    Registry,
    Scorer,
    Selector,
    register_scorer,
)
from dataflex_verl.core.config import validate_compat


# ---------------- Registry ----------------

def test_build_filters_unknown_kwargs():
    reg = Registry()

    @reg.register("scorer", "toy")
    class Toy:
        def __init__(self, alpha=1, beta=2):
            self.alpha = alpha
            self.beta = beta

    # merged cfg+runtime includes an unknown key 'gamma' -> must be dropped
    obj = reg.build("scorer", "toy", runtime={"alpha": 10}, cfg={"beta": 20, "gamma": 99})
    assert obj.alpha == 10  # runtime wins
    assert obj.beta == 20   # from cfg
    assert not hasattr(obj, "gamma")


def test_build_runtime_overrides_cfg():
    reg = Registry()

    @reg.register("scorer", "toy")
    class Toy:
        def __init__(self, x=0):
            self.x = x

    obj = reg.build("scorer", "toy", runtime={"x": 5}, cfg={"x": 1})
    assert obj.x == 5


def test_build_passes_all_when_var_keyword():
    reg = Registry()

    @reg.register("scorer", "greedy")
    class Greedy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    obj = reg.build("scorer", "greedy", runtime={"a": 1}, cfg={"b": 2})
    assert obj.kwargs == {"b": 2, "a": 1}


def test_duplicate_registration_raises():
    reg = Registry()

    @reg.register("scorer", "dup")
    class A:
        pass

    with pytest.raises(ValueError, match="already registered"):
        @reg.register("scorer", "dup")
        class B:
            pass


def test_get_missing_lists_available():
    reg = Registry()

    @reg.register("scorer", "known")
    class A:
        pass

    with pytest.raises(KeyError, match="known"):
        reg.get("scorer", "unknown")


# ---------------- Scorer contract ----------------

def test_scorer_rejects_bad_timing():
    with pytest.raises(ValueError, match="timing"):
        class Bad(Scorer):
            timing = "whenever"

            def score(self, batch, step_id, **ctx):
                return None


def test_scorer_rejects_bad_granularity():
    with pytest.raises(ValueError, match="granularity"):
        class Bad(Scorer):
            granularity = "galaxy"

            def score(self, batch, step_id, **ctx):
                return None


def test_scorer_valid_subclass():
    class Good(Scorer):
        requires = ["token_level_scores"]
        timing = "post_reward"
        granularity = "prompt"

        def score(self, batch, step_id, **ctx):
            return batch

    g = Good()
    assert g.requires == ["token_level_scores"]
    assert g.score(42, 0) == 42


# ---------------- Compatibility validation ----------------

def test_validate_compat_rejects_groups_under_ppo():
    class GroupScorer(Scorer):
        needs_groups = True

        def score(self, batch, step_id, **ctx):
            return None

    with pytest.raises(ValueError, match="group"):
        validate_compat(GroupScorer(), adv_estimator="gae")


def test_validate_compat_allows_groups_under_grpo():
    class GroupScorer(Scorer):
        needs_groups = True

        def score(self, batch, step_id, **ctx):
            return None

    validate_compat(GroupScorer(), adv_estimator="grpo")  # no raise


def test_validate_compat_non_group_scorer_any_algo():
    class Plain(Scorer):
        def score(self, batch, step_id, **ctx):
            return None

    validate_compat(Plain(), adv_estimator="gae")  # no raise


# ---------------- module-level register helpers ----------------

def test_module_register_helpers_share_registry():
    @register_scorer("mod_level")
    class S(Scorer):
        def score(self, batch, step_id, **ctx):
            return 1

    assert "mod_level" in REGISTRY.list("scorer")
