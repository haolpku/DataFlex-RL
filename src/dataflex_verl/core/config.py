"""Component config loading + compatibility validation.

load_component: read a component's params block from a components.yaml, mirroring
DataFlex's utils/load_component.py but framework-agnostic.

validate_compat: the "validate once at mount time" check — given a Scorer and the
active algorithm descriptor, reject incompatible combinations (e.g. a needs_groups
scorer under PPO+GAE) instead of writing per-algorithm variants.
"""

from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

# advantage estimators that produce per-group structure (uid-based)
GROUP_ADV_ESTIMATORS = {"grpo", "grpo_passk", "rloo", "opo", "gdpo"}


def load_component(kind: str, cfg_file: str, name: str, runtime_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load params for a named component from a YAML file.

    Expected layout::

        scorers:
          reward_difficulty:
            name: reward_difficulty
            params: {...}
    """
    if yaml is None:
        raise ImportError("PyYAML is required for load_component; pip install pyyaml")
    with open(cfg_file, "r") as f:
        data = yaml.safe_load(f) or {}
    section = data.get(kind, {})
    entry = section.get(name, {})
    params = dict(entry.get("params", {}))
    runtime_vars = runtime_vars or {}
    # substitute ${var} placeholders in string values
    for k, v in list(params.items()):
        if isinstance(v, str):
            for var, val in runtime_vars.items():
                v = v.replace("${" + var + "}", str(val))
            params[k] = v
    return params


def validate_compat(scorer, adv_estimator: Optional[str] = None) -> None:
    """Raise if the scorer is incompatible with the active algorithm.

    This is the single check that replaces N per-algorithm code copies.
    """
    if getattr(scorer, "needs_groups", False):
        est = (adv_estimator or "").lower()
        if est and est not in GROUP_ADV_ESTIMATORS:
            raise ValueError(
                f"Scorer {type(scorer).__name__} needs group structure (needs_groups=True) "
                f"but adv_estimator={adv_estimator!r} is not group-based "
                f"(expected one of {sorted(GROUP_ADV_ESTIMATORS)})."
            )


# fields that only exist when verl on-policy distillation is running
_TEACHER_FIELDS = {"teacher_logprobs", "teacher_log_probs"}


def validate_opd_compat(scorer, mechanism: str, distillation: Optional[Dict[str, Any]] = None) -> None:
    """Validate DataFlex + verl on-policy-distillation (OPD) combinations at mount time.

    A scorer that reads a teacher field (distill_*) needs OPD enabled. And because
    verl's GKD path (``use_policy_gradient=false``) backprops the distillation loss
    directly WITHOUT multiplying ``rollout_is_weights`` (see
    verl/trainer/distillation/losses.py), reweight/select — which act purely through
    that weight field — silently no-op under GKD. Mix is unaffected (it changes the
    sampling distribution, not the loss). So:

      - teacher-dependent scorer + OPD disabled            -> error (no teacher field).
      - teacher-dependent scorer + reweight/select + GKD   -> error (weights ignored).
      - teacher-dependent scorer + mix                     -> allowed under any mode.
    """
    needs_teacher = bool(_TEACHER_FIELDS & set(getattr(scorer, "requires", [])))
    if not needs_teacher:
        return

    distillation = distillation or {}
    enabled = bool(distillation.get("enabled", False))
    if not enabled:
        raise ValueError(
            f"Scorer {type(scorer).__name__} reads a teacher field "
            f"({sorted(_TEACHER_FIELDS & set(scorer.requires))}) but distillation.enabled is not set. "
            "Enable verl on-policy distillation (distillation.enabled=true) so the teacher "
            "scores each rollout, or use a non-teacher scorer."
        )

    if mechanism in ("reweight", "select"):
        loss_cfg = distillation.get("distillation_loss", {}) or {}
        use_pg = loss_cfg.get("use_policy_gradient", False)
        if not use_pg:
            raise ValueError(
                f"DataFlex '{mechanism}' with a teacher scorer requires PG OPD "
                "(distillation.distillation_loss.use_policy_gradient=true). Under GKD "
                "(use_policy_gradient=false) verl backprops the distillation loss directly "
                "and ignores rollout_is_weights, so reweight/select would silently no-op. "
                "Use PG OPD, or switch mechanism to 'mix' (which does not depend on the loss)."
            )
