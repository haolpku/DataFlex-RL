# Research: algorithms mergeable into DataFlex-RL

Survey of RL data-scheduling algorithms (2022–2026) that map onto DataFlex-RL's
`Scorer` (signal → score) + `Actuator` (score → action) design, without forking
per RL algorithm and — where possible — without extra model forward passes.

Three mechanism-aligned reports:
- [`01_selection.md`](01_selection.md) — Selectors (drop/keep samples)
- [`02_reweighting.md`](02_reweighting.md) — Reweighters (per-sample/-token loss weights)
- [`03_mixture.md`](03_mixture.md) — Mixers (domain sampling proportions)
- [`04_what_beats_baseline.md`](04_what_beats_baseline.md) — **which algorithms have empirical evidence of beating GRPO baseline** (small-model focus) + how our own 4-run result maps to the literature

Each entry lists: signal · rule · granularity · group-dependence · verl implementation
difficulty. Full sources at the end of each file.

---

## What we already have

| Mechanism | Implemented |
|---|---|
| Scorer | `reward_difficulty`, `advantage_magnitude`, `group_solve_rate` |
| Reweighter | `softmax`, `difficulty_band` |
| Selector | `threshold_band`, `topk_fraction` |
| Mixer | `reward_gap`, `static` |

Notably, **DAPO dynamic sampling ≈ `group_solve_rate` + `threshold_band(0,1)`** — we
effectively already have it (minus the resample-to-refill loop).

---

## Prioritized roadmap (highest ROI first)

### P0 — cheap, high-value, pure batch fields
1. **Online Difficulty Filtering** (Selector) — `group_solve_rate` + band `[0.2,0.8]`; a config away. arXiv:2504.03380
2. **Advantage Reweighting (AR)** (Reweighter) — `w_t=α·π_θ+(1−α)` damps low-prob tokens; cleanest `rollout_is_weights` fit, strong reported gains. arXiv:2505.12929
3. **PF-PPO `pow`** (Reweighter) — `w=reward**p`; complements verl's resampling-based `use_pf_ppo`. arXiv:2409.06957
4. **DUMP UCB Mixer** — UCB over per-domain `|advantage|`; adds the exploration `reward_gap` lacks. arXiv:2504.09710
5. **TSCL learning-progress Mixer** — proportions by reward *slope*, a new signal axis vs level. arXiv:1707.00183

### P1 — modest effort, clear value
6. **GFPO** (Selector) — per-group top-k by length or reward/length → concise reasoning. arXiv:2508.09726
7. **PODS** (Selector) — max-variance within-group subset. arXiv:2504.13818
8. **PER-on-|advantage|** (Reweighter) — `w=|A|**α`; we already have the scorer. arXiv:1511.05952
9. **RWR/AWR / GRPO-LEAD / ODSW** (Reweighters) — exp-tilt / logistic / group-accuracy bump. arXiv:1910.00177, 2504.09696
10. **EXP3/ODM Mixer** — adversarial non-stationary bandit; also yields DoReMi's minimax update. arXiv:2312.02406

### P2 — needs new plumbing or token-entropy field
11. **Beyond-80/20 high-entropy token masking** (Selector/Reweighter) — needs per-token entropy field. arXiv:2506.01939
12. **GRESO** (Selector) — pre-rollout skip; genuinely *saves rollout cost* but needs stateful pre-rollout path (replay-buffer layer). arXiv:2506.02177
13. **PLR staleness** (Mixer add-on) — anti-starvation layer atop any Mixer. arXiv:2010.03934

### P3 — heavy / lower priority
14. **RLEP** (Selector) — external replay buffer of verified-correct trajectories. arXiv:2507.07451
15. **GSPO / GEPO** (Reweighter) — sequence-level IS; faithful versions are custom losses. arXiv:2507.18071

---

## Two architectural moves that unlock many at once

1. **Pluggable Mixer strategy** over a Scorer emitting per-domain
   `{score, count, staleness, window_series}` → softmax / UCB / EXP3 / Thompson /
   minimax back-ends become interchangeable (covers #4, #5, #10, DoReMi, PLR).
   Use sliding-window / discounted variants (SW-UCB, D-UCB) since RL reward is
   non-stationary.
2. **Expose per-token entropy** as a batch field → unlocks the entropy-based
   selectors/reweighters (#11, GRPO-S) which several 2025 papers report as high-impact.

---

## Cross-cutting notes
- **"weight 0 ≠ drop":** hard-filter methods (DAPO, PF-PPO `max_min`) are honestly
  Selectors — a zeroed sample still paid its rollout cost. Only pre-rollout selection
  (GRESO, replay-buffer filtering) actually saves generation.
- **Group-only vs universal:** anything keyed on `uid`/pass-rate needs a group
  estimator (GRPO/RLOO); reward-/advantage-/entropy-level methods work for any
  estimator. `needs_groups` already encodes this and is validated at mount time.
- **No double-counting the PPO ratio:** GSPO/TIS touch the ratio already inside
  `pg_losses` — apply as deliberate corrections, not a second ratio.
- **Always mean-normalize weights** (`_normalize_mean_one`) to preserve effective LR.
