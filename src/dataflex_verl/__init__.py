"""dataflex_verl: DataFlex data-scheduling for verl RL training.

Zero-fork plugin: importing this package registers DataFlex components into
verl's open registries (policy loss / trainer). See DESIGN_dataflex_verl.md.
"""

from .core import (
    REGISTRY,
    Scorer,
    Selector,
    Reweighter,
    Mixer,
    register_scorer,
    register_selector,
    register_reweighter,
    register_mixer,
)

__version__ = "0.0.1"


def register_all() -> None:
    """Register all DataFlex components into verl's registries.

    Called via the ``verl.plugins`` entry point at install time, or manually
    with ``import dataflex_verl; dataflex_verl.register_all()``.

    Imports the trainer/replay-buffer modules, whose @register_trainer decorators
    run against verl's registries. Scorer/actuator classes register into our own
    REGISTRY on import (done here too so they're available for build_from_config).
    """
    global _REGISTERED
    if _REGISTERED:
        return
    # our own registry (framework-agnostic; safe without verl)
    from . import scorers, reweighters, selectors, mixers  # noqa: F401
    # verl-coupled: registers trainers into verl's TRAINER_REGISTRY. Requires verl.
    from . import trainer  # noqa: F401
    from . import replay_buffer  # noqa: F401
    _REGISTERED = True


_REGISTERED = False

__all__ = [
    "REGISTRY",
    "Scorer",
    "Selector",
    "Reweighter",
    "Mixer",
    "register_scorer",
    "register_selector",
    "register_reweighter",
    "register_mixer",
    "register_all",
    "__version__",
]
