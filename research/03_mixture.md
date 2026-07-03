# RL Data-Mixture / Domain-Weighting Algorithms — mergeable into DataFlex-RL

Scope: proportion-update rules for a retrospective Mixer that accumulates per-domain
reward/advantage over a window and sets future sampling proportions. Our existing
Mixer is `reward_gap` (softmax over `max_reward − domain_reward`, favoring lagging
domains) with warmup + sliding window; the custom replay buffer
(`DataFlexMixReplayBuffer`) applies the proportions pre-rollout.

Existing code: `src/dataflex_verl/mixers.py` (`RewardGapMixer`, `StaticMixer`,
`DomainStatsTracker`), `replay_buffer.py`.

Update-rule families: **softmax** (Boltzmann over a per-domain score), **bandit**
(EXP3/UCB/Thompson), **minimax/DRO** (worst-case reweighting), **learning-progress**
(slope of reward curve).

---

## Tier 1 — directly fits our signals (reward/advantage stats, no extra forward pass)

### 1. DUMP — UCB bandit over |advantage| ⭐ best single addition
- **Source:** arXiv:2504.09710.
- **Signal:** per-domain `E[|advantage|]` (learnability) + visit counts.
- **Rule:** `UCB(d)=mean|Â_d| + sqrt(2·ln(N+1)/(n_d+1))`, softmax → proportions. Exploit high-learnability + explore under-sampled.
- **Origin:** native RL (GRPO), zero adaptation.
- **Difficulty:** **Easy.** Our `reward_gap` is the exploit-only special case; this adds principled exploration. Scorer = running mean |advantage| + counter; Mixer = UCB+softmax.

### 2. RuleReasoner DADS — EWMA reward + softmax
- **Source:** arXiv:2506.08672.
- **Signal:** EWMA of per-domain mean reward.
- **Rule:** proportion ∝ softmax of EWMA lag/deficit — near-identical to our `reward_gap` but EWMA instead of hard window.
- **Difficulty:** **Easy.** One-line change (`s ← β·s+(1−β)·r`). Good A/B vs our sliding window.

### 3. TSCL — learning-progress / reward-slope ⭐ genuinely new axis
- **Source:** arXiv:1707.00183.
- **Signal:** *slope* dR/dt of per-domain reward over the window (NOT level — the key differentiator from `reward_gap`).
- **Rule:** least-squares slope over (step, mean_reward); sample ∝ softmax(|slope|) or ε-greedy. `|slope|` captures both progress and forgetting.
- **Difficulty:** **Easy–Medium.** We already keep the window; add closed-form slope (~5 lines). Decide signed vs `|slope|`.

### 4. VCRL — group-reward-variance difficulty
- **Source:** arXiv 2509 (Variance-based Curriculum RL; verify exact id).
- **Signal:** within-group (per-uid) reward variance, aggregated per domain.
- **Rule:** proportion ∝ softmax(mean group-reward variance) — high variance = p≈0.5 = max GRPO gradient.
- **Difficulty:** **Easy.** (For GRPO, group variance ≈ |advantage|, so cousin of DUMP's signal — worth both.)

---

## Tier 2 — classic bandits (pluggable Mixer back-ends)

### 5. ODM — Online Data Mixing (EXP3)
- **Source:** arXiv:2312.02406.
- **Signal:** per-domain reward (orig. loss; substitute reward/|advantage|).
- **Rule (EXP3):** `p_d=(1−γ)w_d/Σw+γ/K`; `w_d *= exp(γ·r̂_d/(K·p_d))`. Non-stationary-friendly, regret bounds.
- **Difficulty:** **Easy–Medium.** Importance-weighting + reward normalization need care. Adversarial-bandit Mixer distinct from UCB/softmax.

### 6. EXP3 / UCB1 / Thompson family
- Foundation of #1/#3/#5. **Use sliding-window / discounted variants (SW-UCB, D-UCB, EXP3.S)** — RL reward is non-stationary, plain UCB assumes stationarity.
- **Difficulty:** **Easy.** These are interchangeable back-ends → **build one Scorer interface (per-domain scalar + count) + pluggable Mixer strategy.** Highest-leverage architectural move.

---

## Tier 3 — curriculum-RL literature

### 7. PLR — Prioritized Level Replay (domains as levels)
- **Source:** arXiv:2010.03934.
- **Signal:** mean |advantage| per domain + *staleness* (steps since last sampled).
- **Rule:** `P=(1−ρ)·rank-softmax(score) + ρ·staleness`. Rank-based (scale-robust) + explicit anti-starvation.
- **Difficulty:** **Medium.** Staleness is a cheap anti-starvation add-on usable atop any Tier-1 Mixer.

### 8. RECAP — convergence-rate priority
- **Source:** arXiv:2510.21978. Up-weight slow-converging/unstable domains (slope + volatility). **Medium**, overlaps TSCL+VCRL; add only if forgetting is a concrete problem.

---

## Tier 4 — minimax / DRO

### 9. DoReMi (Group-DRO minimax)
- **Source:** arXiv:2305.10429.
- **Orig signal:** per-domain excess loss vs reference model (**needs ref model** — avoid under no-extra-pass constraint).
- **Online-RL adaptation:** replace excess loss with reward deficit `(max_reward − reward_d)` → exponentiated-weights, **≈ our `reward_gap`**. DoReMi's real contribution to us = the *minimax framing* + *multiplicative mirror-ascent* update instead of plain softmax.
- **Difficulty:** **Medium.** Add as a "minimax mirror-ascent" Mixer variant to contrast with softmax reward_gap.

### 10. GRAPE / Group-DRO rate-of-improvement
- arXiv (May 2025). ≈ TSCL-slope + DoReMi-minimax. Low marginal value once #3+#9 exist.

---

## Not recommended (out of scope)
- **DARS** (arXiv:2508.13755): reallocates *rollouts per problem by difficulty*, not domain proportions; needs pre-rollout difficulty estimate.
- **CHORD** (arXiv:2510.21978 line): on/off-policy token weighting, not mixing.
- **Data Mixing Agent** (2025): separate RL policy + eval feedback — heavy, needs eval passes.

---

## Recommendation
Refactor to a **pluggable Mixer strategy** over a common Scorer emitting per-domain
`{score, count, staleness, window_series}`. Signals (all free): mean reward, mean
|advantage|, group-reward variance, reward slope, visit count, staleness.

Priority Mixer strategies to add:
- **P0 UCB/DUMP** (#1) — adds the exploration `reward_gap` lacks.
- **P0 TSCL slope** (#3) — new signal axis (slope not level).
- **P1 EXP3/ODM** (#5) — non-stationary-robust; gives DoReMi's minimax update for free.
- **P1 VCRL variance** (#4) — cheap sweet-spot difficulty.
- **P2 EWMA/DADS** (#2) — trivial A/B variant of current Mixer.
- **P2 PLR staleness** (#7) — anti-starvation layer atop any strategy.

Use sliding-window/discounted bandit variants throughout (non-stationary reward).

## Sources
- https://arxiv.org/abs/2504.09710 (DUMP)
- https://arxiv.org/abs/2506.08672 (RuleReasoner / DADS)
- https://arxiv.org/abs/1707.00183 (TSCL)
- https://arxiv.org/abs/2312.02406 (ODM)
- https://arxiv.org/abs/2010.03934 (Prioritized Level Replay)
- https://arxiv.org/abs/2510.21978 (RECAP)
- https://arxiv.org/abs/2305.10429 (DoReMi)
- https://arxiv.org/abs/2508.13755 (DARS — out of scope)
- VCRL / GRAPE: 2025 arXiv, exact ids to be confirmed before citing in code.
