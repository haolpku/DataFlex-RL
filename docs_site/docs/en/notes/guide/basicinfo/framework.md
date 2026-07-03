---
title: Framework Design
icon: material-symbols:auto-transmission-sharp
createTime: 2026/07/03 10:00:00
permalink: /en/guide/basicinfo/framework/
---

# Framework Design

## Overview

DataFlex-RL is a **zero-fork plugin** for the [verl](https://github.com/volcengine/verl)
RL training library. It intelligently schedules training data during RL fine-tuning,
supporting **dynamic sample reweighting**, **sample selection**, and **domain ratio
adjustment** to improve training efficiency and final model performance.

### Design Philosophy

The core philosophy is **data-centric intelligent scheduling for RL**: instead of
treating every rollout equally, let the model dynamically adjust how each sample
contributes — based on signals it already produces during training (reward, advantage,
log-probs, group structure). Crucially, DataFlex-RL does this **without modifying verl
source** and **without extra model forward passes** — it reads batch fields that RL
training computes anyway.

## Core Architecture

### Two-layer decomposition: Scorer + Actuator

The central idea is to split every data-scheduling algorithm into two reusable layers:

```
┌──────────────────────────────────────────────────────────────────────┐
│                         verl RL training loop                          │
│           rollout → reward → advantage → actor/critic update           │
├──────────────────────────────────────────────────────────────────────┤
│  DataFlex-RL plugin (registered into verl's open registries)           │
│                                                                        │
│   ┌─────────────────────────────────────────────────────────────┐    │
│   │  Scorer   (signal → score)   —  SHARED across mechanisms      │    │
│   │  reads only declared batch fields (reward / advantage / uid   │    │
│   │  / log_probs); returns a per-sample or per-token score        │    │
│   └─────────────────────────────────────────────────────────────┘    │
│                                │                                       │
│         ┌──────────────────────┼──────────────────────┐               │
│         ▼                      ▼                      ▼               │
│   ┌───────────┐          ┌───────────┐          ┌───────────┐         │
│   │ Reweighter│          │ Selector  │          │  Mixer    │         │
│   │ score →   │          │ score →   │          │ score →   │         │
│   │ weights   │          │ keep idx  │          │ domain %  │         │
│   └───────────┘          └───────────┘          └───────────┘         │
│  (per-token loss)      (drop samples)      (future sampling)          │
└──────────────────────────────────────────────────────────────────────┘
```

**Why split this way?** Because a data-scheduling algorithm has two independent
questions:
1. *What signal do we score on?* (reward, advantage, group pass-rate, token prob…)
2. *What do we do with the score?* (weight the loss, drop the sample, re-mix domains)

The same **Scorer** feeds all three mechanisms, so scoring logic is written **once**,
never duplicated per mechanism or per RL algorithm.

### Scorer

A Scorer maps a training signal to a score. It **declares its requirements** so the
host can validate compatibility with the active algorithm *once at mount time*:

```python
class Scorer(ABC):
    requires: list[str]      # batch fields it reads, e.g. ["advantages", "response_mask"]
    timing: str              # "pre_rollout" | "post_reward" | "post_advantage" | "in_loss"
    granularity: str         # "domain" | "prompt" | "response" | "token"
    needs_groups: bool       # True → only valid for group estimators (GRPO/RLOO)

    def score(self, batch, step_id, **ctx): ...   # → (bs,) or (bs, L)
```

`needs_groups` is the key to **not forking per algorithm**: a group-only scorer (e.g.
group solve-rate) is rejected at startup on PPO+GAE, while reward-/advantage-level
scorers work for any estimator. One config check replaces N per-algorithm code copies.

### Actuator

Three mechanisms consume the score. They differ in **mount point**, **output type**,
and **cost semantics** — which is exactly why they cannot be merged into one:

| Mechanism | Output | Mount point | When |
|---|---|---|---|
| **Reweight** | per-token weights → `rollout_is_weights` | trainer `_compute_advantage` | every step, in-loop |
| **Select** | 0/1 mask (drops gradient) | trainer `_compute_advantage` | every step, in-loop |
| **Mix** | domain proportions → sampler | custom `ReplayBuffer` + trainer | periodic, pre-rollout |

**A crucial RL-specific distinction:** "weight 0" (Reweight) ≠ "drop" (Select). A
zeroed sample still paid its rollout cost; only *pre-rollout* selection actually saves
generation. Reweight and Select share the post-advantage hook because both reduce to
per-token weights that verl's vanilla policy loss already multiplies in — **no custom
policy loss is needed**. Mix is retrospective and per-domain: it accumulates each
domain's mean reward and steers *future* sampling, so it needs a warmup phase.

## How it mounts into verl (zero fork)

verl v1 exposes open registries that DataFlex-RL registers into on `import`:

1. **`register_trainer`** → selected by `config.trainer.v1.trainer_mode`
   (`dataflex_sync` for reweight/select, `dataflex_mix_sync` for mix).
2. **Custom replay-buffer sampler** → `config.trainer.v1.sampler.custom_sampler`
   (for Mix's domain-proportional sampling).
3. **`rollout_is_weights`** → verl's existing per-token weight hook, which the vanilla
   PPO loss multiplies into `pg_losses` before aggregation (this is what Reweight and
   Select write to).

### Zero-config auto-registration

Because verl scans the `verl.plugins` entry-point group on `import verl`, merely
installing `dataflex_verl` (whose entry point targets the `dataflex_verl.autoload`
module) registers the trainers **in the driver and in every Ray worker** — no manual
import, no `PYTHONPATH`. Add a `config.dataflex` block to verl's YAML and run verl's
normal entrypoint.

## Component Hierarchy

1. **Host Layer (verl)** — rollout, reward, advantage, actor/critic workers, TransferQueue data plane.
2. **Trainer Layer (DataFlex-RL trainers)** — subclass verl's v1 sync trainer; override
   `_compute_advantage` (reweight/select) or the sampler + stats hooks (mix). No `fit()` rewrite.
3. **Component Layer** — `Scorer` + `Actuator` (Reweighter / Selector / Mixer).
4. **Registry** — framework-agnostic `core/`: registration, config building, compatibility validation.

**Key feature:** DataFlex-RL doesn't add layers on top of verl — it **registers into
verl's existing hooks**, keeping the RL loop intact while enhancing data scheduling.

## Timing model: SFT discrete-phase vs RL in-loop

A common misconception is that RL data scheduling is the SFT "select a batch → train →
re-select" discrete-phase loop. It is **not**:

- **Reweight / Select** are **in-loop**: every step, after reward/advantage, the scorer
  reads the current batch and the actuator acts on it immediately.
- **Mix** is the only mechanism that keeps a periodic rhythm — it accumulates per-domain
  stats over a sliding window and updates future sampling proportions (retrospective,
  needs a warmup cold-start).

## Next

- [Installation](../install/)
- [Reweighter](/en/guide/reweighter/quickstart/) · [Selector](/en/guide/selector/quickstart/) · [Mixer](/en/guide/mixer/quickstart/)
- To add your own algorithm, see each mechanism's **tutorial** page.
