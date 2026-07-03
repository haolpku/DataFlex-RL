---
title: Introduction
icon: mdi:tooltip-text-outline
createTime: 2026/07/03 10:00:00
permalink: /en/guide/basicinfo/intro/
---

# DataFlex-RL

**Data scheduling (Select · Mix · Reweight) for RL fine-tuning of LLMs — a zero-fork
plugin for [verl](https://github.com/volcengine/verl).**

DataFlex-RL brings DataFlex's data-centric training philosophy to reinforcement
learning (PPO / GRPO / RLVR). It plugs into verl's open registries without modifying
verl source: install both, add a few YAML lines, and run verl's normal entrypoint.

```bash
pip install verl
pip install dataflex_verl
```

## Why data scheduling for RL?

In RL fine-tuning, not every rollout is equally useful. A prompt whose responses are
all correct (or all wrong) carries no gradient signal; a low-probability token can
dominate the update; one data domain may already be mastered while another lags.
DataFlex-RL lets the model **dynamically decide, from signals it already produces
during training, which data to emphasize, drop, or sample more of** — without a
second reward model or extra forward passes.

The RL setting is actually *richer* than SFT for data scheduling: besides loss, every
step produces **reward, advantage, per-token log-probs, and group structure (uid)** —
a much stronger set of signals to schedule on.

## Three mechanisms

| Mechanism | What it does | Mount point |
|---|---|---|
| **Reweight** | per-token / per-sample loss weights | trainer, after advantage |
| **Select** | drop samples (zero their gradient) | trainer, after advantage |
| **Mix** | dynamic domain sampling proportions | custom replay buffer, pre-rollout |

All three are expressed through one design: a shared **Scorer** (`signal → score`)
feeding a mechanism-specific **Actuator** (`score → action`). See
[Framework Design](../framework/) for the rationale.

## Relationship to DataFlex

[DataFlex](https://github.com/OpenDCAI/DataFlex) is the original data-centric training
system for **SFT**, built on LLaMA-Factory. DataFlex-RL is its **RL counterpart**: the
same Select / Mix / Reweight vocabulary and the same Scorer/Actuator decomposition,
re-grounded on verl's RL training loop and signals. The two share design DNA but are
independent packages (different host frameworks).

## Verified

All three mechanisms train end-to-end on 8×GPU (Qwen2.5, GSM8K / dapo-math, verl v1)
and evaluate through the Qwen2.5-Math harness. See the repository's `experiments/` for
runnable comparison scripts and results.

## Next

- [Framework Design](../framework/) — the Scorer / Actuator architecture and verl mount points
- [Installation](../install/) — install + zero-config auto-registration
- Then dive into a mechanism: [Reweighter](/en/guide/reweighter/quickstart/) ·
  [Selector](/en/guide/selector/quickstart/) · [Mixer](/en/guide/mixer/quickstart/)
