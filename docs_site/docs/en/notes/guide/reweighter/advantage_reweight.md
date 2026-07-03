---
title: Advantage Reweighting
icon: material-symbols:balance
createTime: 2026/07/03 10:00:00
permalink: /en/guide/reweighter/advantage_reweight/
---

# Advantage Reweighting (low-probability token damping)

**Source:** *Do Not Let Low-Probability Tokens Over-Dominate in RL for LLMs*
([arXiv:2505.12929](https://arxiv.org/abs/2505.12929)).

## Motivation

In policy-gradient RL, a token's gradient magnitude scales inversely with its
probability. Low-probability tokens therefore produce **outsized gradients** and can
dominate the update, destabilizing training. Advantage Reweighting (AR) damps them
with a per-token weight that grows with the token's probability.

## Rule

For each token *t* with policy probability `π_θ(t)`:

$$
w_t = \alpha \cdot \pi_\theta(t) + (1 - \alpha)
$$

- Low-prob tokens (`π→0`) get weight near `1−α` (damped).
- High-prob tokens (`π→1`) get weight near `1`.
- Weights are mean-normalized over valid tokens to preserve the effective LR.

`α ∈ [0, 1]` controls damping strength; `α=0` recovers the unweighted baseline.

## In DataFlex-RL

AR is **token-granularity**: it uses the `token_prob` scorer (per-token
`π_θ = exp(old_log_prob)`, forward-pass-free) and the `advantage_reweight` reweighter,
which returns a `(bs, L)` weight matrix directly.

```yaml
trainer: {v1: {trainer_mode: dataflex_sync}}
dataflex:
  mechanism: reweight
  scorer:   {name: token_prob}
  actuator: {name: advantage_reweight, params: {alpha: 0.5}}
  warmup_step: 0
```

CLI:

```bash
+dataflex.mechanism=reweight \
+dataflex.scorer.name=token_prob \
+dataflex.actuator.name=advantage_reweight \
+dataflex.actuator.params.alpha=0.5
```

## Why it's a strong pick for small models

Among reweighting methods surveyed, AR has the **most direct small-model evidence**:
the paper reports gains at **3B and 7B** (e.g. +46% relative on Knights-and-Knaves
logic, positive on math), and it needs only `old_log_probs` — no extra forward pass,
no reward model. Gains tend to grow in **later training stages**.

## Signal / rule / granularity

| Property | Value |
|---|---|
| Signal | per-token `π_θ` (from `old_log_probs`) |
| Rule | `w = α·π + (1−α)`, mean-normalized |
| Granularity | token |
| Needs groups | no (any advantage estimator) |
| Extra forward pass | no |
