# RL Sample/Loss Reweighting Algorithms — mergeable into DataFlex-RL

Scope: methods producing per-sample or per-token weights that multiply into
`pg_losses` before aggregation. verl already exposes `rollout_is_weights` (per-token
multiply in the vanilla policy loss) — our `DataFlexSyncTrainer` writes it via
`response_to_nested`. "Fits the hook cleanly" = a nonneg multiplier, mean-normalized
to ~1 (our `_normalize_mean_one`) to preserve step size.

Existing code: `src/dataflex_verl/reweighters.py` (`SoftmaxReweighter`,
`DifficultyBandReweighter`).

Note: verl also ships `compute_pf_ppo_reweight_data` (`algorithm.use_pf_ppo`), but
that does **weighted resampling with replacement**, not a per-token multiply — so
re-expressing PF-PPO as a *Reweighter* is a genuinely useful, cleaner variant.

---

## Tier 1 — easy, well-used, exact hook fit (no extra forward pass)

### PF-PPO (pow / max_min / max_random)
- **Source:** arXiv:2409.06957 (Policy Filtration in RLHF). Origin of verl's `pf_ppo`.
- **Signal:** per-sample reward (`token_level_scores` summed).
- **Weight:** `pow`: `w=score**p`; `max_min`/`max_random`: keep group max/min (or max+random), others 0.
- **Granularity:** per-sample. **Groups:** `max_min`/`max_random` need `uid`; `pow` doesn't.
- **Difficulty:** **Easy.** `pow` is a one-line reweighter; group variants need uid argmax/min.

### Advantage Reweighting (AR) — low-prob token damping
- **Source:** arXiv:2505.12929 ("Do Not Let Low-Probability Tokens Over-Dominate").
- **Signal:** per-token `π_θ` (from `log_probs`).
- **Weight:** `w_t = α·exp(log_prob) + (1−α)` — down-weights low-prob tokens.
- **Granularity:** per-token. **Groups:** no.
- **Difficulty:** **Easy.** Pure batch fields; one of the cleanest hook fits. (+35.9% GRPO on K&K logic in-paper.)

### RWR / AWR / RAFT (reward/advantage-weighted regression)
- **Source:** AWR arXiv:1910.00177; RAFT arXiv:2304.06767.
- **Signal:** reward (RWR/RAFT) or advantage (AWR).
- **Weight:** exponential tilt `exp(A/β)`, clipped + normalized. RAFT = binary top-1-per-prompt.
- **Granularity:** per-sample. **Groups:** none (AWR works with any advantage).
- **Difficulty:** **Easy.** Clip aggressively (interacts with ratio×adv already in pg_losses).

### GRPO-LEAD — difficulty-aware logistic reweight
- **Source:** arXiv:2504.09696.
- **Signal:** per-prompt accuracy `ρ_q=#correct/#rollouts` (reward+uid).
- **Weight:** logistic `w(ρ_q)=A+(B−A)/(1+exp[k(ρ_q−ρ0)])`; hard prompts → larger weight.
- **Granularity:** per-sample (per-prompt). **Groups:** yes.
- **Difficulty:** **Easy–Medium.** Group-reduce reward→accuracy→logistic.

### ODSW — medium-difficulty bump (group-accuracy version of our band)
- **Source:** VL-Cogito (ODSW); relates to DAPO dynamic sampling.
- **Signal:** per-prompt group accuracy.
- **Weight:** soft triangular/Gaussian bump peaked at ρ=0.5.
- **Granularity:** per-sample. **Groups:** yes.
- **Difficulty:** **Easy.** Like our `DifficultyBandReweighter` but keyed on *group accuracy* not raw-score quantiles — worth adding as a variant.

### PER-on-|advantage| / focal-style hard-example weighting
- **Source:** PER arXiv:1511.05952; Focal Loss arXiv:1708.02002.
- **Signal:** `|advantage|`.
- **Weight:** `w=|A|**α`, mean-normalized (up-weight high-surprise samples).
- **Granularity:** per-sample. **Groups:** no.
- **Difficulty:** **Easy.** (Without actual non-uniform sampling this is just hard-example up-weighting; true prioritized *sampling* is a Mixer/Selector concern.)

### Token-level Truncated Importance Sampling (TIS) — verl-native
- **Source:** verl `compute_rollout_importance_weights` (PR #2953/#3694).
- **Signal:** `log_probs` vs `old_log_prob`/rollout log-probs.
- **Weight:** `w_t=min(exp(logπ_θ−logπ_behav), c)`.
- **Granularity:** per-token. **Groups:** no.
- **Difficulty:** **Easy** — literally the hook's native purpose. Beware double-counting the PPO ratio already in pg_losses.

---

## Tier 2 — medium / borderline the hook

### High-entropy token masking (Beyond 80/20 as a reweighter)
- **Source:** arXiv:2506.01939. Weight 1 for top-k% entropy tokens, else 0. Needs per-token entropy field. **Medium.**

### GRPO-S / GTPO — entropy-scaled token weighting
- **Source:** arXiv:2508.04349. Boost high-entropy tokens in correct seqs, penalize confident-wrong tokens. GRPO-S (per-seq avg entropy) fits the hook; GTPO (per-position group norm) is heavier. **Medium.**

### GSPO — sequence-level importance ratio (as reweighter)
- **Source:** arXiv:2507.18071 (Qwen3). Faithful GSPO is a custom loss, but a reweighter `w_tok=(seq geo ratio)/(per-token ratio)` approximates it into the hook. **Medium**, algebra-fiddly.

### LUFFY policy shaping
- **Source:** arXiv:2504.14945. `f(π_θ)=π_θ/(π_θ+γ)` on off-policy tokens; needs off-policy trajectories tagged in-batch. **Medium.**

---

## Better done at advantage/aggregation stage (not per-token multiply)
- **DIET** (separate reward/penalty advantage normalization) — arXiv, edits advantage composition.
- **Dr.GRPO / DAPO / ΔL length-debias** (arXiv:2503.20783 / 2503.14476 / 2509.07558) — these are loss-aggregation fixes (`loss_agg_mode`), expressible as length weights but risk double-correcting.
- **GEPO / full GTPO** — group-expectation IS, heavier.

---

## Recommended next adds (by ROI)
- **P0:** Advantage Reweighting (AR) — cleanest hook fit, strong reported gains, pure `log_probs`.
- **P0:** PF-PPO `pow` — trivial, well-known, complements resampling variant verl already has.
- **P1:** RWR/AWR exp-tilt on advantage; GRPO-LEAD logistic; ODSW group-accuracy bump (variant of our existing band reweighter).
- **P1:** PER-on-|advantage| (we already have the `advantage_magnitude` scorer — just add a power-weight reweighter).
- **P2:** high-entropy token masking, GRPO-S (need per-token entropy field).

Implementation notes: (1) all P0/P1 need only `token_level_scores`/`advantages`/`uid`/`log_probs`/`response_mask` — no extra forward; (2) always mean-normalize; (3) watch double-counting the PPO ratio (GSPO/TIS); (4) hard-mask methods (PF-PPO `max_min`) are honestly Selectors — weight-0 still pays rollout cost.

## Sources
- https://arxiv.org/abs/2409.06957 (PF-PPO) · https://github.com/swtheing/PF-PPO-RLHF
- https://arxiv.org/abs/1910.00177 (AWR) · https://arxiv.org/abs/2304.06767 (RAFT)
- https://arxiv.org/abs/2504.09696 (GRPO-LEAD)
- https://arxiv.org/abs/2505.12929 (Advantage Reweighting / Lopti)
- https://arxiv.org/abs/2506.01939 (Beyond 80/20) · https://arxiv.org/abs/2508.04349 (GTPO/GRPO-S)
- https://arxiv.org/abs/2507.18071 (GSPO) · https://arxiv.org/abs/2504.14945 (LUFFY)
- https://arxiv.org/abs/2503.20783 (Dr.GRPO) · https://arxiv.org/abs/2509.07558 (ΔL Norm)
- https://arxiv.org/abs/1511.05952 (PER) · https://arxiv.org/abs/1708.02002 (Focal Loss)
- https://arxiv.org/abs/2503.14476 (DAPO) · https://github.com/volcengine/verl (TIS/RolloutIS)
