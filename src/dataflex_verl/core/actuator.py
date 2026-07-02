"""Actuator: the mechanism layer (three-way split).

Select / Reweight / Mix share a Scorer but differ in *mount point*, *output type*,
and *cost semantics*, so they cannot be merged into one interface:

  Selector   : score -> List[int] indices   (changes batch membership; can skip rollout)
  Reweighter : score -> weights tensor       (multiplies pg_losses before aggregation)
  Mixer      : score -> domain proportions   (changes future sampling distribution)

In RL specifically, "weight 0" (Reweighter) != "drop" (Selector): the dropped
sample never pays the rollout cost. That distinction is why the split is required.
"""

from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np


class Actuator(ABC):
    """Base for the three mechanisms. Holds a reference to its Scorer."""

    def __init__(self, scorer=None, **kwargs):
        self.scorer = scorer

    @abstractmethod
    def act(self, scores: Any, batch: Any, **ctx):
        ...


class Selector(Actuator):
    """score -> indices to keep (drops prompts and all their rollouts)."""

    @abstractmethod
    def act(self, scores: Any, batch: Any, **ctx) -> List[int]:
        ...


class Reweighter(Actuator):
    """score -> per-sample/-token weights, multiplied into the policy loss."""

    @abstractmethod
    def act(self, scores: Any, batch: Any, **ctx):
        ...


class Mixer(Actuator):
    """score -> domain proportions for the sampler (retrospective, per-domain)."""

    @abstractmethod
    def act(self, scores: Any, batch: Any, **ctx) -> "np.ndarray":
        ...
