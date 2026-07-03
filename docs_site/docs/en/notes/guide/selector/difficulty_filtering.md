---
title: Online Difficulty Filtering
icon: solar:filter-bold
createTime: 2026/07/03 10:00:00
permalink: /en/guide/selector/difficulty_filtering/
---

# Online Difficulty Filtering

**Source:** *Online Difficulty Filtering for Reasoning-Oriented RL*
([arXiv:2504.03380](https://arxiv.org/abs/2504.03380)).

## Motivation

The RL learning signal is maximized when a prompt's pass-rate is near **0.5**: a prompt
the model always solves (rate=1) or never solves (rate=0) has zero group-advantage and
contributes no gradient. Online Difficulty Filtering keeps prompts whose current
pass-rate sits in a **balanced band** around 0.5, focusing the update where it is most
informative.

## Rule

Keep prompt *x* iff its batch-estimated pass-rate satisfies:

$$
T_\text{low} \le p(x) \le T_\text{high}
$$

with e.g. `T_low = 0.2, T_high = 0.8`. This **generalizes DAPO dynamic sampling**,
which is the degenerate band `(0, 1)` that only excludes exactly-0 and exactly-1.

## In DataFlex-RL

No new code — it's the `group_solve_rate` scorer + `threshold_band` selector with the
0.2/0.8 band:

```yaml
trainer: {v1: {trainer_mode: dataflex_sync}}
dataflex:
  mechanism: select
  scorer:   {name: group_solve_rate, params: {success_threshold: 0.5}}
  actuator: {name: threshold_band, params: {low: 0.2, high: 0.8}}
  warmup_step: 0
```

## Evidence & conditions

Reported **+10% AIME / +4% average** over GRPO across five math benchmarks; the
mechanism is scale-agnostic. **Important condition:** gains require a dataset with a
**wide difficulty spread**. On a uniformly-easy set (e.g. plain GSM8K) there is little
to filter, and aggressive filtering can thin the effective batch and *hurt* — so pair
it with harder / more varied data (e.g. dapo-math).

## Signal / rule / granularity

| Property | Value |
|---|---|
| Signal | per-group pass-rate (from `rm_scores` + `uid`) |
| Rule | keep pass-rate ∈ [0.2, 0.8] |
| Granularity | prompt (group) |
| Needs groups | yes (GRPO/RLOO/…) |
| Extra forward pass | no |
