"""verl v1 trainer subclass that applies a DataFlex Reweighter or Selector.

Registered as ``dataflex_sync`` (extends the built-in ``sync`` trainer). It hooks
``_compute_advantage``: after verl computes advantages/returns and stores them in the
TransferQueue, we read the signal fields back, run scorer->actuator, and write a
``rollout_is_weights`` field back into the queue. verl's vanilla policy loss then
multiplies pg_losses by it — no custom policy loss required.

Two mechanisms share this hook because both reduce to per-token weights:
  - reweight : soft weights in [0, inf), mean-normalized.
  - select   : hard 0/1 weights (dropped samples contribute zero gradient).

NOTE on select semantics: filtering *here* (post-advantage) removes a sample's
gradient contribution but does NOT save its rollout cost — the generation already
happened. The rollout-saving, pre-rollout variant belongs at the replay-buffer /
dataset layer; this post-advantage select is the reward/advantage-driven variant
(e.g. DAPO-style dropping of zero-signal groups after seeing their reward).

Mixture (M3) is handled separately at the replay-buffer layer (replay_buffer.py).

This module imports verl and is only loaded by register_all() at runtime.
"""

import logging
import os
import uuid

import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# verl imports — only available in a verl environment
import numpy as np  # noqa: E402
import transfer_queue as tq  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from verl.protocol import DataProto  # noqa: E402
from verl.trainer.ppo.v1.trainer_base import register_trainer  # noqa: E402
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync  # noqa: E402
from verl.utils import tensordict_utils as tu  # noqa: E402
from verl.workers.utils.padding import response_to_nested  # noqa: E402

from .build import build_from_config  # noqa: E402
from .mixers import DomainStatsTracker  # noqa: E402
from . import replay_buffer as _rb  # noqa: E402


@register_trainer("dataflex_sync")
class DataFlexSyncTrainer(PPOTrainerSync):
    """Sync PPO trainer with DataFlex reweight/select injected after advantage."""

    _SUPPORTED = {"reweight", "select"}

    def __init__(self, config):
        super().__init__(config)
        self._df_scorer = None
        self._df_actuator = None
        self._df_meta = None
        df_cfg = config.get("dataflex", None)
        if df_cfg is not None and df_cfg.get("mechanism", None) in self._SUPPORTED:
            self._df_scorer, self._df_actuator, self._df_meta = build_from_config(
                df_cfg, adv_estimator=str(config.algorithm.adv_estimator),
                distillation=config.get("distillation", None),
            )
            logger.info(
                f"[DataFlex] {self._df_meta['mechanism']} enabled: "
                f"scorer={type(self._df_scorer).__name__}, "
                f"actuator={type(self._df_actuator).__name__}, warmup={self._df_meta['warmup_step']}"
            )

    def _weights_from_actuator(self, scores, dp, n) -> torch.Tensor:
        """Turn actuator output into a per-sample weight vector of length n."""
        out = self._df_actuator.act(scores, dp)
        if self._df_meta["mechanism"] == "select":
            # actuator returns a list of kept indices -> 0/1 mask
            keep = torch.zeros(n, dtype=torch.float32)
            idx = torch.as_tensor(list(out), dtype=torch.long)
            if idx.numel() > 0:
                keep[idx] = 1.0
            return keep
        # reweight: actuator returns a weight tensor
        return torch.as_tensor(out, dtype=torch.float32).flatten()

    def _compute_advantage(self, batch, metrics):
        # let verl compute advantages/returns and persist them
        batch = super()._compute_advantage(batch, metrics)

        if self._df_actuator is None:
            return batch
        if self.global_steps < self._df_meta["warmup_step"]:
            return batch

        # read back the tensor fields our scorer needs (+ what we broadcast over).
        # Mirror verl's _compute_advantage: fetch uid alongside, then pop it out of the
        # padded tensordict into non_tensor_batch (raw TQ uid is a LinkedList, not a tensor).
        # Selectors may be group-based (GFPO / PODS group by uid), so fetch uid for any
        # select mechanism too, not only when the scorer declares it.
        want_uid = ("uid" in self._df_scorer.requires) or (self._df_meta["mechanism"] == "select")
        need = sorted((set(self._df_scorer.requires) | {"response_mask"} | ({"uid"} if want_uid else set())))
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=need)
        response_mask = data["response_mask"]
        dp = DataProto(batch=data.to_padded_tensor())
        if want_uid:
            dp.non_tensor_batch["uid"] = np.array(dp.batch.pop("uid").tolist(), dtype=object)

        n = dp.batch["response_mask"].shape[0]
        pad_mask = dp.batch["response_mask"].float()
        token_level = getattr(self._df_scorer, "granularity", "prompt") == "token"

        with torch.no_grad():
            scores = self._df_scorer.score(dp, self.global_steps)          # (bs,) or (bs,L)
            if token_level:
                # token-granularity reweighting (e.g. Advantage Reweighting): the
                # actuator returns a per-token weight matrix (bs, L); no broadcast.
                w_tok = self._df_actuator.act(scores, dp).float() * pad_mask
                # summary metric over valid tokens only
                denom = pad_mask.sum().clamp(min=1.0)
                metrics["dataflex/weight_mean"] = float((w_tok * pad_mask).sum() / denom)
                metrics["dataflex/weight_std"] = float(w_tok[pad_mask.bool()].std())
            else:
                weights = self._weights_from_actuator(scores, dp, n).float()  # (bs,)
                w_tok = weights.view(-1, 1) * pad_mask                        # (bs, L)
                metrics["dataflex/weight_mean"] = float(weights.mean())
                metrics["dataflex/weight_std"] = float(weights.std())
                if self._df_meta["mechanism"] == "select":
                    metrics["dataflex/kept_frac"] = float((weights > 0).float().mean())

        # persist rollout_is_weights so the actor loss picks it up
        out = {"rollout_is_weights": response_to_nested(w_tok, response_mask)}
        out = TensorDict(out, batch_size=len(batch))
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=out)
        return batch



@register_trainer("dataflex_mix_sync")
class DataFlexMixSyncTrainer(PPOTrainerSync):
    """Sync PPO trainer with DataFlex domain mixture (pre-rollout sampling control).

    Pairs with DataFlexMixReplayBuffer (set via trainer.v1.sampler.custom_sampler).
    This trainer:
      1. tags each prompt with its ``data_source`` so the buffer can bucket by domain;
      2. after reward/advantage, accumulates each domain's mean reward into a tracker;
      3. after warmup, every ``update_step`` steps, runs the Mixer to update the shared
         proportions the buffer samples from.
    """

    def __init__(self, config):
        super().__init__(config)
        self._df_scorer = None
        self._df_mixer = None
        self._df_meta = None
        self._df_tracker = None
        self._df_domains = None
        self._df_domain_key = "domain"
        self._df_last_domains = None
        df_cfg = config.get("dataflex", None)
        if df_cfg is not None and df_cfg.get("mechanism", None) == "mix":
            domains = list(df_cfg.get("domains", []))
            assert domains, "config.dataflex.domains must list the domain names for mix"
            self._df_domains = domains
            self._df_domain_key = str(df_cfg.get("domain_key", "domain"))
            self._df_scorer, self._df_mixer, self._df_meta = build_from_config(
                df_cfg, adv_estimator=str(config.algorithm.adv_estimator),
                runtime={"domains": domains},
                distillation=config.get("distillation", None),
            )
            self._df_tracker = DomainStatsTracker(window=int(df_cfg.get("window", 50)))
            # start from static prior (uniform) so the buffer has something during warmup
            _rb.set_proportions(domains, [1.0 / len(domains)] * len(domains))
            logger.info(
                f"[DataFlex] mix enabled: domains={domains}, "
                f"mixer={type(self._df_mixer).__name__}, warmup={self._df_meta['warmup_step']}"
            )

    def _add_batch_to_generate(self):
        """Same as base, but stamps each prompt tag with its domain.

        Domain is read from the dataset column named by ``config.dataflex.domain_key``
        (default "domain"), falling back to "data_source". Kept separate from
        "data_source" so the real data_source still drives verl's reward function.
        The domain is written into the prompt tag under "data_source" (the key the
        DataFlexMixReplayBuffer buckets on).
        """
        if self._df_mixer is None:
            return super()._add_batch_to_generate()

        try:
            if self.train_dataloader_it is None:
                self.train_dataloader_it = iter(self.train_dataloader)
            batch_dict = next(self.train_dataloader_it)
        except StopIteration:
            self.train_dataloader_it = iter(self.train_dataloader)
            batch_dict = next(self.train_dataloader_it)

        n = len(batch_dict["raw_prompt"])
        batch_dict["uid"] = np.array([str(uuid.uuid4()) for _ in range(n)], dtype=object)
        batch = tu.get_tensordict(batch_dict)
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)

        domains = batch_dict.get(self._df_domain_key, None)
        if domains is None:
            domains = batch_dict.get("data_source", None)
        if domains is None:
            domains = ["default"] * n
        self._df_last_domains = [str(domains[i]) for i in range(n)]
        tags = [
            {"is_prompt": True, "status": "pending", "global_steps": self.global_steps,
             "data_source": self._df_last_domains[i]}
            for i in range(n)
        ]
        tq.kv_batch_put(keys=list(batch["uid"]), partition_id="train", tags=tags)
        self.agent_loop_manager.generate_sequences(batch)

    def _per_seq_signal(self, dp):
        """Per-sequence domain signal from the configured scorer.

        Generalizes the old hard-coded "mean rm_scores" so a Mixer can be driven by
        ANY scorer's signal. ``reward_difficulty`` reproduces the reward level exactly;
        ``distill_gap`` (OPD) yields per-domain teacher-student divergence — mixing by
        "how much the teacher can still teach in this domain". A token-granularity
        scorer is mean-aggregated over valid tokens to a per-seq scalar.
        """
        scores = self._df_scorer.score(dp, self.global_steps)   # (bs,) or (bs, L)
        if getattr(self._df_scorer, "granularity", "prompt") == "token":
            mask = dp.batch["response_mask"].to(scores.dtype)
            denom = mask.sum(dim=-1).clamp(min=1.0)
            return (scores * mask).sum(dim=-1) / denom
        return scores.flatten()

    def _compute_advantage(self, batch, metrics):
        batch = super()._compute_advantage(batch, metrics)
        if self._df_mixer is None:
            return batch

        # fetch exactly the fields the scorer needs (+ response_mask) so the mix
        # signal can be reward, teacher-divergence, etc. — whatever the scorer reads.
        need = sorted(set(self._df_scorer.requires) | {"response_mask"})
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=need,
        )
        dp = DataProto(batch=data.to_padded_tensor())
        with torch.no_grad():
            per_seq = self._per_seq_signal(dp)  # (bs,)

        # domain per row from prompt tags
        domains_per_row = []
        for tag in batch.tags:
            domains_per_row.append(str(tag.get("data_source", "default")))
        for d, r in zip(domains_per_row, per_seq.tolist()):
            self._df_tracker.update(d, r)

        # after warmup, refresh proportions periodically
        if (
            self.global_steps >= self._df_meta["warmup_step"]
            and self.global_steps % max(1, self._df_meta["update_step"]) == 0
        ):
            stats = {d: self._df_tracker.mean(d) for d in self._df_domains}
            props = self._df_mixer.act(stats, dp)
            _rb.set_proportions(self._df_domains, list(props))
            for d, p in zip(self._df_domains, props):
                metrics[f"dataflex/prop_{d}"] = float(p)
                metrics[f"dataflex/signal_{d}"] = float(stats[d])
        return batch
