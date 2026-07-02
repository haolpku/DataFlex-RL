"""Framework-agnostic data-scheduling primitives (internal to dataflex_verl).

Two-layer design:
  - Scorer:  signal -> per-sample/-token/-domain score   (SHARED across mechanisms)
  - Actuator: score -> action                             (Selector / Reweighter / Mixer)

See DESIGN_dataflex_verl.md for rationale. Kept as an internal `core` module
(not a separate pip package) since dataflex_verl is the sole consumer.
"""

from .registry import (
    REGISTRY,
    Registry,
    register_mixer,
    register_reweighter,
    register_scorer,
    register_selector,
)
from .scorer import Scorer
from .actuator import Actuator, Mixer, Reweighter, Selector
from .config import load_component, validate_compat

__all__ = [
    "REGISTRY",
    "Registry",
    "register_scorer",
    "register_selector",
    "register_reweighter",
    "register_mixer",
    "Scorer",
    "Actuator",
    "Selector",
    "Reweighter",
    "Mixer",
    "load_component",
    "validate_compat",
]
