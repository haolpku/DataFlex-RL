"""Concrete scorers for verl RL signals.

Each scorer reads only the batch fields named in ``requires`` and returns a score
tensor. Scorers are shared across Selector / Reweighter / Mixer actuators.

The batch passed to ``score`` is a verl ``DataProto`` (has ``.batch`` TensorDict and
``.non_tensor_batch`` dict). Scorers here operate at the point *after advantage
computation*, where these fields exist:
  - advantages, returns            : (bs, response_length)
  - token_level_scores/rewards     : (bs, response_length)   (outcome reward per token)
  - response_mask                  : (bs, response_length)
  - non_tensor_batch["uid"]        : (bs,) group id (GRPO/RLOO/...)
"""

import numpy as np
import torch

from .core.registry import register_scorer
from .core.scorer import Scorer


def _seq_sum(mat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum a (bs, L) tensor over valid tokens -> (bs,)."""
    return (mat * mask.to(mat.dtype)).sum(dim=-1)


# verl persists the outcome reward in the TransferQueue under "rm_scores";
# "token_level_scores" is only an in-DataProto alias set during _compute_advantage.
# Read whichever is present so scorers work both in real verl runs and offline tests.
_REWARD_KEYS = ("rm_scores", "token_level_scores")


def _reward_tensor(batch):
    for k in _REWARD_KEYS:
        if k in batch.batch:
            return batch.batch[k]
    raise KeyError(f"none of {_REWARD_KEYS} found in batch; have {list(batch.batch.keys())}")


@register_scorer("advantage_magnitude")
class AdvantageMagnitudeScorer(Scorer):
    """|advantage| aggregated per sequence. Large |adv| = strong learning signal.

    Works for any advantage estimator (field is standardized), hence needs_groups=False.
    """

    requires = ["advantages", "response_mask"]
    timing = "post_advantage"
    granularity = "prompt"
    needs_groups = False

    def __init__(self, agg: str = "mean", **kwargs):
        super().__init__(**kwargs)
        assert agg in ("mean", "sum", "max")
        self.agg = agg

    def score(self, batch, step_id, **ctx):
        adv = batch.batch["advantages"]
        mask = batch.batch["response_mask"].to(adv.dtype)
        a = adv.abs() * mask
        if self.agg == "sum":
            return a.sum(dim=-1)
        if self.agg == "max":
            return a.amax(dim=-1)
        denom = mask.sum(dim=-1).clamp(min=1.0)
        return a.sum(dim=-1) / denom


@register_scorer("reward_difficulty")
class RewardDifficultyScorer(Scorer):
    """Outcome reward per sequence -> a difficulty signal.

    reward high  = easy (model already solves it)
    reward low   = hard
    Returns the raw per-sequence outcome reward (bs,); actuators decide how to use it
    (e.g. Reweighter up-weights mid-difficulty, Selector filters extremes).
    """

    requires = ["rm_scores", "response_mask"]
    timing = "post_reward"
    granularity = "prompt"
    needs_groups = False

    def score(self, batch, step_id, **ctx):
        # outcome reward spread over tokens; sum -> per-seq scalar
        scores = _reward_tensor(batch)
        mask = batch.batch["response_mask"]
        return _seq_sum(scores, mask)


@register_scorer("group_solve_rate")
class GroupSolveRateScorer(Scorer):
    """Per-group solve rate (mean outcome reward within a uid group).

    Group-based: only meaningful for GRPO/RLOO/... where each prompt has N rollouts
    sharing a uid. Returns a per-sample tensor equal to that sample's group solve rate,
    so a Selector can drop all-solved (rate==1) / all-failed (rate==0) groups (DAPO-style).
    """

    requires = ["rm_scores", "response_mask", "uid"]
    timing = "post_reward"
    granularity = "prompt"
    needs_groups = True

    def __init__(self, success_threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.success_threshold = success_threshold

    def score(self, batch, step_id, **ctx):
        scores = _reward_tensor(batch)
        mask = batch.batch["response_mask"]
        per_seq = _seq_sum(scores, mask)  # (bs,)
        success = (per_seq > self.success_threshold).float()

        uid = batch.non_tensor_batch["uid"]  # (bs,) object array
        uid = np.asarray(uid)
        out = torch.zeros_like(per_seq)
        for g in np.unique(uid):
            idx = np.where(uid == g)[0]
            rate = success[idx].mean()
            out[idx] = rate
        return out


@register_scorer("token_prob")
class TokenProbScorer(Scorer):
    """Per-token policy probability π_θ(t) = exp(old_log_prob), shape (bs, L).

    The signal for Advantage Reweighting (arXiv:2505.12929): low-probability tokens
    produce outsized gradients and over-dominate the update. Returning the per-token
    probability lets a token-granularity reweighter damp them.

    We use ``old_log_probs`` (the log-prob under the policy that generated the rollout,
    persisted in the TransferQueue) as a stable, forward-pass-free proxy for π_θ.
    """

    requires = ["old_log_probs", "response_mask"]
    timing = "post_advantage"
    granularity = "token"
    needs_groups = False

    def score(self, batch, step_id, **ctx):
        logp = batch.batch["old_log_probs"]           # (bs, L)
        return logp.exp()                             # π in (0, 1]


# --- On-Policy Distillation (OPD) signal ---------------------------------------
#
# When verl runs with distillation.enabled=true, the teacher scores every student
# rollout during the Agent Loop and persists ``teacher_logprobs`` (shape (bs, L, 1|K))
# into the batch. By the time our _compute_advantage hook fires, this field is present
# alongside the student's ``old_log_probs`` — so we can form a teacher-student divergence
# signal *for free* (no extra forward pass in DataFlex; the teacher pass already happened).
#
# The per-token reverse-KL term (Thinking Machines "On-Policy Distillation", 2025;
# verl PG OPD uses the same k1/k3 estimators) is
#     k_t = log pi_theta(y_t) - log nu(y_t)          # student_logp - teacher_logp
# k_t > 0 means the student is MORE confident than the teacher at the sampled token
# (over-committing) — the direction OPD pushes down. Both scorers below key on |k_t|
# or (k_t)_+ depending on how the paired actuator uses it.
#
# ``teacher_logprobs`` may carry a trailing top-k axis (K>=1); the sampled-token logprob
# is column 0 (verl packs the sampled token first), matching the single-sample estimator.

_TEACHER_KEYS = ("teacher_logprobs", "teacher_log_probs")


def _teacher_logp(batch):
    """Fetch the teacher per-token logprob at the sampled token -> (bs, L).

    Accepts (bs, L) or (bs, L, K); takes column 0 of the last axis when 3-D.
    Aligns to the response segment length: if teacher width exceeds student
    ``old_log_probs`` width (verl may pad teacher to prompt+response), slice
    the trailing ``response_len`` columns.

    Raises a clear error (naming the OPD switch) if the field is absent.
    """
    b = batch.batch
    for k in _TEACHER_KEYS:
        if k in b:
            t = b[k]
            if t.dim() == 3:
                t = t[..., 0]
            # Align to response length: teacher may be prompt+response wide.
            if "response_mask" in b:
                resp_len = b["response_mask"].shape[-1]
                if t.shape[-1] != resp_len:
                    if t.shape[-1] > resp_len:
                        t = t[..., -resp_len:]
                    else:
                        # rare: pad right with zeros
                        pad = resp_len - t.shape[-1]
                        t = torch.nn.functional.pad(t, (0, pad))
            return t
    raise KeyError(
        f"none of {_TEACHER_KEYS} found in batch (have {list(b.keys())}); "
        "distill_* scorers require verl on-policy distillation "
        "(set distillation.enabled=true so the teacher scores each rollout)."
    )


@register_scorer("distill_kl")
class DistillKLScorer(Scorer):
    """Per-token teacher-student divergence for OPD-driven data scheduling.

    k_t = old_log_probs - teacher_logp  (reverse-KL term at the sampled token).
    Returns a (bs, L) matrix; pair with a token-granularity reweighter to focus the
    gradient where student and teacher disagree most.

    ``mode``:
      - "signed"  : raw k_t (can be +/-).
      - "abs"     : |k_t|            (default; magnitude of disagreement).
      - "pos"     : max(k_t, 0)      (only where student over-commits vs teacher).
    """

    requires = ["old_log_probs", "teacher_logprobs", "response_mask"]
    timing = "post_advantage"
    granularity = "token"
    needs_groups = False

    def __init__(self, mode: str = "abs", **kwargs):
        super().__init__(**kwargs)
        assert mode in ("signed", "abs", "pos")
        self.mode = mode

    def score(self, batch, step_id, **ctx):
        student = batch.batch["old_log_probs"].float()   # (bs, L)
        teacher = _teacher_logp(batch).float()           # (bs, L)
        k = student - teacher                            # reverse-KL term
        if self.mode == "abs":
            return k.abs()
        if self.mode == "pos":
            return k.clamp(min=0.0)
        return k


@register_scorer("distill_gap")
class DistillGapScorer(Scorer):
    """Per-sequence teacher-student divergence (aggregate of distill_kl).

    D_i = sum_t |k_t| / L_i  over valid tokens -> (bs,). A per-sample "how much is
    still learnable from the teacher" signal; feed a Selector (keep high-gap samples)
    or a Mixer (proportion domains by mean gap). ``mode`` matches DistillKLScorer.
    """

    requires = ["old_log_probs", "teacher_logprobs", "response_mask"]
    timing = "post_advantage"
    granularity = "prompt"
    needs_groups = False

    def __init__(self, mode: str = "abs", **kwargs):
        super().__init__(**kwargs)
        assert mode in ("signed", "abs", "pos")
        self.mode = mode

    def score(self, batch, step_id, **ctx):
        student = batch.batch["old_log_probs"].float()
        teacher = _teacher_logp(batch).float()
        k = student - teacher
        if self.mode == "abs":
            k = k.abs()
        elif self.mode == "pos":
            k = k.clamp(min=0.0)
        mask = batch.batch["response_mask"].to(k.dtype)
        denom = mask.sum(dim=-1).clamp(min=1.0)
        return (k * mask).sum(dim=-1) / denom            # (bs,)
