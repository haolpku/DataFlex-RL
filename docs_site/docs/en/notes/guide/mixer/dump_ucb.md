---
title: DUMP (UCB Mixer)
icon: solar:chart-2-outline
createTime: 2026/07/03 10:00:00
permalink: /en/guide/mixer/dump_ucb/
---

# DUMP — UCB Bandit Mixer

**Source:** *DUMP: Automated Distribution-Level Curriculum Learning for RL-based LLM
Post-training* ([arXiv:2504.09710](https://arxiv.org/abs/2504.09710)).

## Motivation

The default `reward_gap` mixer is **exploit-only**: it always favors the
lowest-reward domain, which can over-invest in a domain that is simply too hard
(unsolvable → no learning). DUMP treats each domain as a **bandit arm** whose value is
its **learnability** — the mean absolute advantage `E[|Â|]` — and adds a UCB
**exploration bonus** for under-sampled domains. `|advantage|` self-regulates: it peaks
at mid-difficulty and is near zero for both mastered and impossible domains.

## Rule

For each domain *d* with mean |advantage| `v_d` and visit count `n_d` (total `N`):

$$
\text{UCB}(d) = v_d + c\sqrt{\frac{2\ln(N+1)}{n_d + 1}}, \qquad
p_d = \mathrm{softmax}\big(\text{UCB}(d) / T\big)
$$

with a floor to avoid starving any domain. The first term exploits high-learnability
domains; the second explores rarely-sampled ones.

## In DataFlex-RL

```yaml
trainer:
  v1:
    trainer_mode: dataflex_mix_sync
    sampler:
      custom_sampler: {path: pkg://dataflex_verl.replay_buffer, name: DataFlexMixReplayBuffer}
dataflex:
  mechanism: mix
  domains: [math, code, logic]
  scorer:   {name: advantage_magnitude}      # per-domain |advantage|
  actuator: {name: dump_ucb, params: {temperature: 1.0, c: 1.0, floor: 0.05}}
  warmup_step: 5
  update_step: 5
```

## Evidence & conditions

DUMP beats uniform sampling on multi-domain logic/math and auto-induces an easy→hard
curriculum. Because the `|advantage|` signal is exactly what GRPO already computes, it
is **near-zero overhead**. As with all mixers, it needs **≥3 heterogeneous domains** to
show real value; with 2 domains a tuned static ratio is competitive.

## Signal / rule / granularity

| Property | Value |
|---|---|
| Signal | per-domain mean \|advantage\| + visit counts |
| Rule | UCB + softmax over domains |
| Granularity | domain |
| Needs groups | no (any estimator that yields advantages) |
| Extra forward pass | no |

## Related mixers worth adding

The survey (`research/03_mixture.md` in the repo) flags **TSCL** (reward-slope /
learning-progress), **ODM/EXP3** (adversarial bandit), and **VCRL** (group-reward
variance) as next candidates. A clean way to add them is a **pluggable Mixer back-end**
over a Scorer emitting `{score, count, staleness}` — softmax / UCB / EXP3 / Thompson
become interchangeable strategies.
