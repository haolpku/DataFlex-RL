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
