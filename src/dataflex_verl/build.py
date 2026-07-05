"""Build scorer + actuator from a `config.dataflex` block, with compatibility check.

The `config.dataflex` block (added to verl's YAML) looks like::

    dataflex:
      mechanism: reweight        # reweight | select | mix
      scorer:
        name: advantage_magnitude
        params: {agg: mean}
      actuator:
        name: softmax
        params: {temperature: 1.0}
      warmup_step: 0             # steps before the mechanism activates
      update_step: 1             # apply every N steps (mix only; reweight/select every step)

This module is import-light so it can be unit-tested without verl.
"""

from .core.config import validate_compat, validate_opd_compat
from .core.registry import REGISTRY

# ensure component classes are registered
from . import scorers as _scorers  # noqa: F401
from . import reweighters as _reweighters  # noqa: F401
from . import selectors as _selectors  # noqa: F401
from . import mixers as _mixers  # noqa: F401

_ACTUATOR_KIND = {"reweight": "reweighter", "select": "selector", "mix": "mixer"}


def _as_dict(x):
    """OmegaConf DictConfig or plain dict -> plain dict."""
    if x is None:
        return {}
    if hasattr(x, "items") and not isinstance(x, dict):
        try:
            from omegaconf import OmegaConf

            return OmegaConf.to_container(x, resolve=True)
        except Exception:
            return dict(x)
    return dict(x)


def build_from_config(dataflex_cfg, *, adv_estimator=None, runtime=None, distillation=None):
    """Return (scorer, actuator, meta) built from a `config.dataflex` block.

    ``runtime`` supplies actuator constructor deps not in cfg (e.g. domains list for
    a Mixer). ``adv_estimator`` is used to reject group-only scorers on non-group algos.
    ``distillation`` is verl's ``config.distillation`` block (or None); it lets the
    OPD compatibility check reject teacher scorers when distillation is off, or when
    reweight/select is combined with GKD (which ignores rollout_is_weights).
    """
    cfg = _as_dict(dataflex_cfg)
    runtime = runtime or {}

    mechanism = cfg.get("mechanism")
    if mechanism not in _ACTUATOR_KIND:
        raise ValueError(f"config.dataflex.mechanism must be one of {list(_ACTUATOR_KIND)}, got {mechanism!r}")

    scorer_cfg = _as_dict(cfg.get("scorer"))
    act_cfg = _as_dict(cfg.get("actuator"))

    scorer = REGISTRY.build(
        "scorer", scorer_cfg["name"], runtime={}, cfg=_as_dict(scorer_cfg.get("params"))
    )
    validate_compat(scorer, adv_estimator=adv_estimator)
    validate_opd_compat(scorer, mechanism, distillation=_as_dict(distillation))

    actuator = REGISTRY.build(
        _ACTUATOR_KIND[mechanism],
        act_cfg["name"],
        runtime={"scorer": scorer, **runtime},
        cfg=_as_dict(act_cfg.get("params")),
    )

    meta = {
        "mechanism": mechanism,
        "warmup_step": int(cfg.get("warmup_step", 0)),
        "update_step": int(cfg.get("update_step", 1)),
    }
    return scorer, actuator, meta
