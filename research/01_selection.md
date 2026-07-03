# RL Sample Selection / Dynamic Sampling Algorithms — mergeable into DataFlex-RL

Scope: algorithms that select/drop samples from signals present in a training batch
(reward, advantage, group pass-rate, length, logprob, entropy), mapping onto a
`Scorer` (per-sample/per-group score) + `Selector` (keep/drop rule). Difficulty is
rated for a verl plugin seeing only batch fields (`rm_scores`/`token_level_scores`,
`advantages`, `uid`, `old_log_probs`, `response_mask`).

Related existing code: `src/dataflex_verl/selectors.py` (`ThresholdBandSelector`,
`TopKFractionSelector`), `scorers.py` (`group_solve_rate`, `advantage_magnitude`).

---

## 1. DAPO — Dynamic Sampling (filter all-correct/all-wrong groups)
- **Source:** arXiv:2503.14476 (ByteDance/Tsinghua). verl ships a DAPO recipe.
- **Signal:** per-group reward variance / pass-rate.
- **Rule:** keep group iff `0 < #correct < G` (drop std==0 groups).
- **Granularity:** group (prompt). **Needs GRPO groups:** yes.
- **verl difficulty:** **Easy** — already ≈ our `group_solve_rate` + `threshold_band(0,1)`. The only extra piece is the resample-to-refill loop (lives above the selector). **Status: essentially already implemented.**

## 2. Online Difficulty Filtering (pass-rate band around 0.5)
- **Source:** arXiv:2504.03380.
- **Signal:** per-prompt pass-rate from the batch's rollouts.
- **Rule:** keep iff `T_low ≤ p(x) ≤ T_high` (e.g. 0.2–0.8) — a generalization of DAPO (which only excludes 0 and 1).
- **Granularity:** prompt/group. **Needs groups:** yes.
- **verl difficulty:** **Easy** — our `group_solve_rate` scorer + a two-threshold band selector. Basically a config of what we have (set low/high to 0.2/0.8). **High ROI, trivial add.**

## 3. GRESO — pre-rollout selective rollout
- **Source:** arXiv:2506.02177 ("Act Only When It Pays").
- **Signal:** temporal history of per-prompt reward variance (which prompts were zero-variance in recent cycles).
- **Rule:** probabilistic skip `p_f(x)=1 − p_e·1[z_i≥1]`, auto-tuned to a target zero-variance ratio; acts BEFORE generation → saves rollout cost.
- **Granularity:** prompt (pre-rollout). **Needs groups:** yes (for the signal).
- **verl difficulty:** **Medium–Hard** — needs persistent cross-batch per-prompt state + a pre-rollout hook (replay-buffer layer, like our Mix path), not a stateless post-advantage selector. This is the genuinely rollout-saving variant our DESIGN calls out.

## 4. PODS — max-variance rollout down-sampling
- **Source:** arXiv:2504.13818 ("Not All Rollouts are Useful").
- **Signal:** per-response reward within a group.
- **Rule:** keep the size-n subset with maximum reward spread (binary case: equal #highest + #lowest).
- **Granularity:** response (within group). **Needs groups:** yes.
- **verl difficulty:** **Easy–Medium** — Scorer=per-response reward, Selector=per-group top/bottom pairing; mask dropped responses (advantage=0). A new "within-group subset" selector mode.

## 5. GFPO — Group Filtered Policy Optimization (top-k per group)
- **Source:** arXiv:2508.09726 (Microsoft). "Sample more to think less."
- **Signal:** configurable — response length, reward/length (token efficiency), or reward.
- **Rule:** keep top-k/G by the metric; filtered responses get advantage 0. Adaptive-difficulty variant varies k by group pass-rate.
- **Granularity:** response ranking within group. **Needs groups:** yes.
- **verl difficulty:** **Easy** — Scorer=length or reward/length (from `response_mask` sum + reward), Selector=per-group top-k → advantage 0. Great for concise-reasoning objectives.

## 6. Beyond the 80/20 Rule — high-entropy token selection
- **Source:** arXiv:2506.01939 (Qwen/Tsinghua).
- **Signal:** per-token policy entropy.
- **Rule:** keep top-~20% highest-entropy ("forking") tokens; zero the loss mask elsewhere.
- **Granularity:** token. **Needs groups:** no.
- **verl difficulty:** **Medium** — a token-mask selector over `response_mask`; needs per-token entropy exposed as a batch field (verl computes it for the entropy bonus). Different granularity than our sample/group selectors.

## 7. RLEP — replay of verified-correct trajectories
- **Source:** arXiv:2507.07451 (Kuaishou Klear). Code: github.com/Kwai-Klear/RLEP.
- **Signal:** correctness (only correct trajectories admitted to pool).
- **Rule:** blend a fixed fraction of replayed correct trajectories into each batch.
- **Granularity:** trajectory. **Needs groups:** no.
- **verl difficulty:** **Hard** — needs an external persistent replay buffer + re-injection, beyond a stateless Scorer+Selector.

---

## Quick-reference

| Algo | Signal | Rule | Granularity | Groups | Difficulty | Priority |
|---|---|---|---|---|---|---|
| DAPO dyn. sampling | group variance | drop std==0 | group | yes | Easy | ✅ have it |
| Online Difficulty Filter | group pass-rate | band [0.2,0.8] | prompt | yes | Easy | **P0** |
| GFPO | length / rew-per-tok | top-k/group | response | yes | Easy | **P1** |
| PODS | per-response reward | max-variance subset | response | yes | Easy–Med | P1 |
| \|advantage\| top-k | advantage | top-k / threshold | sample | no | Easy | **P0** (have scorer) |
| GRESO | prompt var history | pre-rollout skip | prompt | yes | Med–Hard | P2 (saves rollout) |
| Beyond 80/20 | token entropy | top-20% tokens | token | no | Med | P2 |
| RLEP | correctness | replay pool | traj | no | Hard | P3 |

**Recommended next adds:** Online Difficulty Filtering (P0, one config away), GFPO
top-k selector (P1), PODS within-group subset (P1). GRESO is the standout for
*rollout-cost saving* but needs the stateful pre-rollout path.

## Sources
- https://arxiv.org/abs/2503.14476 (DAPO)
- https://arxiv.org/abs/2504.03380 (Online Difficulty Filtering)
- https://arxiv.org/abs/2506.02177 (GRESO)
- https://arxiv.org/abs/2504.13818 (PODS)
- https://arxiv.org/abs/2508.09726 (GFPO)
- https://arxiv.org/abs/2506.01939 (Beyond the 80/20 Rule)
- https://arxiv.org/abs/2507.07451 (RLEP) / https://github.com/Kwai-Klear/RLEP
