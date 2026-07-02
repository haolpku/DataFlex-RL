"""DataFlex Mixer replay buffer (M3): domain-proportional sampling.

Plugged in via ``trainer.v1.sampler.custom_sampler.{path,name}``. It overrides
``sample()`` to select prompts according to per-domain proportions that a Mixer
updates from a sliding window of each domain's mean reward.

Cold start: until ``warmup_step`` prompts have been observed, proportions fall back
to the static prior (uniform or configured). This is the retrospective, per-domain,
periodic nature of mixture — it steers *future* sampling, not the current batch.

The per-prompt domain is read from the prompt tag ``data_source`` (verl already
carries dataset ``data_source`` into the batch; a small trainer hook copies it into
the prompt tag — see DataFlexMixSyncTrainer).

NOTE: this reads the accumulated domain reward from a module-level tracker updated by
the trainer's on_step_end (reward stats are only known after rollout+reward). The
buffer itself only needs the current proportions + each prompt's domain tag.
"""

import logging
import os
import random

import transfer_queue as tq
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer

from .build import build_from_config

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# shared between the custom buffer and the trainer that updates domain stats
_MIX_STATE = {"proportions": None, "domains": None}


def set_proportions(domains, proportions):
    _MIX_STATE["domains"] = list(domains)
    _MIX_STATE["proportions"] = list(proportions)


def get_state():
    return _MIX_STATE


class DataFlexMixReplayBuffer(ReplayBuffer):
    """Samples prompts to match domain proportions from the shared mix state."""

    def _domain_of(self, tag) -> str:
        return str(tag.get("data_source", "default"))

    def sample(self, global_steps: int, partition_id: str, batch_size: int) -> KVBatchMeta:
        self._sync_metadata_from_transfer_queue()
        while not self._has_enough_samples(global_steps, partition_id, batch_size):
            import time

            time.sleep(self.poll_interval)
            self._sync_metadata_from_transfer_queue()

        finished = self.finished_keys[partition_id]
        failure = self.failure_keys[partition_id]
        prompt_steps = self.prompt_global_steps[partition_id]
        sampleable = sorted(finished.union(failure), key=lambda k: prompt_steps.get(k, 0))

        proportions = _MIX_STATE.get("proportions")
        domains = _MIX_STATE.get("domains")

        if proportions is None or domains is None:
            # cold start / no mixer -> fall back to default oldest-first behavior
            selected_prompt_uids = sampleable[:batch_size]
        else:
            selected_prompt_uids = self._proportional_select(
                sampleable, partition_id, domains, proportions, batch_size
            )
            # robustness: never return fewer than available; top up oldest-first
            if len(selected_prompt_uids) < min(batch_size, len(sampleable)):
                have = set(selected_prompt_uids)
                for uid in sampleable:
                    if len(selected_prompt_uids) >= batch_size:
                        break
                    if uid not in have:
                        selected_prompt_uids.append(uid)
                        have.add(uid)

        tq.kv_clear(partition_id=partition_id, keys=selected_prompt_uids)

        keys, tags = [], []
        selected = set(selected_prompt_uids)
        for key, tag in self.partitions[partition_id].items():
            uid = key.split("_")[0]
            if uid in selected:
                keys.append(key)
                tags.append(tag)
        batch = KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags)
        return self._drop_max_off_policy_samples(global_steps, partition_id, batch)

    def _proportional_select(self, sampleable, partition_id, domains, proportions, batch_size):
        # bucket sampleable prompt uids by domain (prompt tag lives in partitions)
        part = self.partitions[partition_id]
        by_domain = {d: [] for d in domains}
        # map prompt uid -> its prompt tag; prompt key == uid (is_prompt)
        for uid in sampleable:
            tag = part.get(uid, {})
            d = self._domain_of(tag)
            by_domain.setdefault(d, []).append(uid)

        # target counts per domain (largest-remainder rounding)
        targets = {}
        raw = {d: proportions[i] * batch_size for i, d in enumerate(domains)}
        floored = {d: int(v) for d, v in raw.items()}
        remainder = batch_size - sum(floored.values())
        frac_order = sorted(domains, key=lambda d: raw[d] - floored[d], reverse=True)
        for d in frac_order[:remainder]:
            floored[d] += 1
        targets = floored

        selected = []
        leftover_pool = []
        for d in domains:
            pool = by_domain.get(d, [])
            take = min(targets[d], len(pool))
            selected.extend(pool[:take])
            leftover_pool.extend(pool[take:])
        # fill any shortfall (a domain lacked enough prompts) from leftovers
        shortfall = batch_size - len(selected)
        if shortfall > 0:
            selected.extend(leftover_pool[:shortfall])
        return selected[:batch_size]
