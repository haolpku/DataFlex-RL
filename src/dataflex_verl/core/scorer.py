"""Scorer: the SHARED scoring layer.

A Scorer maps a training signal to a score tensor. The same score can feed any
of the three actuators (Selector / Reweighter / Mixer), so scoring logic is
written once, never duplicated per mechanism or per RL algorithm.

Scorers declare their requirements so the host framework can validate
compatibility with the active algorithm *once at mount time*, instead of forking
code per algorithm:

  requires      : batch fields the scorer reads (e.g. ["token_level_scores"])
  timing        : when the signal is available in the pipeline
  granularity   : the natural unit of the score
  needs_groups  : True -> only valid for group-based algos (GRPO/RLOO/GDPO);
                  the host rejects/degrades on PPO+GAE.

These are class attributes with sane defaults; subclasses override as needed.
The batch object is framework-specific (verl DataProto, a dict, ...); the Scorer
only touches the fields named in ``requires``.
"""

from abc import ABC, abstractmethod
from typing import Any, List

# timing stages, ordered by availability in an RL step
TIMINGS = ("pre_rollout", "post_reward", "post_advantage", "in_loss")
GRANULARITIES = ("domain", "prompt", "response", "token")


class Scorer(ABC):
    requires: List[str] = []
    timing: str = "post_reward"
    granularity: str = "prompt"
    needs_groups: bool = False

    def __init__(self, **kwargs):
        # subclasses may consume their own params; extra kwargs ignored by design
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.timing not in TIMINGS:
            raise ValueError(f"{cls.__name__}.timing={cls.timing!r} not in {TIMINGS}")
        if cls.granularity not in GRANULARITIES:
            raise ValueError(f"{cls.__name__}.granularity={cls.granularity!r} not in {GRANULARITIES}")

    @abstractmethod
    def score(self, batch: Any, step_id: int, **ctx):
        """Return a score tensor.

        Shape depends on ``granularity``: (bs,) for prompt/response, (bs, resp_len)
        for token, or (num_domains,) for domain-level aggregates.
        """
        ...
