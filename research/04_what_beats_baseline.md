# Which algorithms are most likely to beat the GRPO baseline?

This report goes beyond "what can we implement" (see `README.md`, `01/02/03_*.md`)
to the sharper question: **which algorithms have real empirical evidence of beating a
plain GRPO/PPO baseline, and under what conditions** — especially for small models
(0.5B–7B) on math reasoning, which is our setting.

Verdict scale: **HIGH / MEDIUM / LOW** = confidence of a *real accuracy gain* over
plain GRPO (not just speed/length wins).

---

## TL;DR — the three best bets

| Mechanism | Method | Small-model gain | Evidence |
|---|---|---|---|
| **Reweight** | **Advantage Reweighting** (低概率 token 抑制, 2505.12929) | **HIGH** | +46% rel on K&K logic, positive on math, validated at **3B & 7B** |
| **Select** | **Online Difficulty Filtering** (pass-rate band, 2504.03380) | **HIGH** | **+10% AIME / +4% avg** over GRPO across 5 math benchmarks |
| **Mix** | **DADS** (per-domain EWMA reward→softmax, 2506.08672) / learnability signal (DUMP/VCRL) | **HIGH (≥3 domains)** | RuleReasoner-4B/8B beat uniform GRPO + static Mix-SFT head-to-head |

These three are the only candidates with **direct, multi-condition, small-model
evidence of accuracy gains**, and all are cheap (batch-field signals, no extra
forward pass). Start here.

---

## Reweighting

| Method | Result (reported) | Scale shown | Verdict |
|---|---|---|---|
| **Advantage Reweighting + Lopti** (2505.12929) | +46.2% rel on Knights-and-Knaves; positive on math; gains grow late in training | **3B, 7B** | **HIGH** — strongest small-model reweighting evidence, cheap |
| **GRPO-LEAD** difficulty logistic (2504.09696) | AIME24 Cons@32 0.80→0.867 at 14B; 7B ablation modest (0.533→0.567), part from bundled length reward | 7B / 14B | **MEDIUM** — real but modest standalone |
| **PF-PPO** (2409.06957) | SOTA on HumanEval/MBPP over PPO/DPO at 7B | 7B | **LOW for RLVR** — designed for *noisy reward models*; verifiable 0/1 rewards lack the noise it targets. Only worth it with a learned RM |
| **Beyond 80/20** high-entropy token mask (2506.01939) | AIME24 +7.7, AIME25 +11 at **32B** | 32B | **LOW at small scale** — explicit scale effect, minimal ≤8B. Large-model-only |

## Selection

| Method | Result (reported) | Scale | Verdict |
|---|---|---|---|
| **Online Difficulty Filtering** (2504.03380) | +10% AIME, +4% avg over GRPO, 5 benchmarks | scale-agnostic mechanism | **HIGH** — best pure-selection accuracy evidence |
| **DAPO dynamic sampling** (2503.14476) | 32B full stack 30→50 AIME; filtering *alone* ≈ efficiency/stability | 32B (component scale-free) | **MEDIUM–HIGH** — near-free, mostly convergence/stability at small scale, not big accuracy |
| **PODS** max-variance (2504.13818) | beats GRPO on GSM8K only | small | **LOW–MEDIUM** — thin evidence, mainly efficiency |
| **GRESO** pre-rollout (2506.02177) | 2–2.4× speedup, accuracy *comparable* to GRPO | 1.5B–7B | **LOW for accuracy, HIGH for efficiency** — same result, faster |
| **GFPO** (2508.09726) | 46–85% shorter output, accuracy flat | 14B | **LOW for accuracy** — it's a length/conciseness method |

## Mixture

| Method | Result | Scale | Verdict |
|---|---|---|---|
| **DADS / RuleReasoner** (2506.08672) | 4B/8B beat uniform GRPO + static Mix-SFT head-to-head, +4–10% OOD | **4B, 8B** | **HIGH (multi-domain)** — best small-model mixture evidence |
| **DUMP** UCB over \|advantage\| (2504.09710) | beats uniform on logic/math, auto easy→hard curriculum | small logic models | **MEDIUM–HIGH** — advantage signal is GRPO-native, near-zero cost |
| **VCRL** variance curriculum (2509.19803) | Qwen3-4B/8B large gains over GRPO/DAPO/GSPO (in-domain difficulty, not cross-domain) | 4B, 8B | **HIGH in-domain / MEDIUM as mixture** — variance is a clean learnability signal |
| **ODM** EXP3 (2312.02406) | ~19% fewer steps, +1.9% MMLU — *pretraining*, not RLVR | 1B pretraining | **MEDIUM (pretrain) / LOW (RLVR transfer)** |
| **DoReMi** (2305.10429) | +6.5% downstream at 8B *pretraining*, needs proxy runs, offline | 280M→8B | reference static baseline, too heavy for online plugin |

---

## Key conditions that gate gains (read before implementing)

1. **Reweighting (Advantage Reweighting):** best small-model bet; gains concentrate
   *late* in training and on structured/logic tasks, more modest on pure math. Cheap.

2. **Selection (Difficulty Filtering):** needs a dataset with **wide difficulty
   spread** — gains come from focusing on pass-rate≈0.5 prompts. On a uniformly-easy
   set (like plain GSM8K) there's little to filter, and aggressive filtering can *thin
   the batch and hurt* — exactly what our own experiment showed (see below).

3. **Mixture:** the big caveat — **dynamic mixing barely helps with 2 domains**
   (reduces to one ratio knob, a tuned static ratio is competitive). It pays off at
   **≥3 domains with heterogeneous difficulty**. Prefer a **learnability signal**
   (reward variance / |advantage|, peaks at mid-difficulty) over raw mean-reward gap,
   which can over-invest in unsolvable domains.

---

## What our own 4-run experiment showed (step 300, 0.5B, GSM8K)

| Run | GSM8K | MATH | AMC23 | mean |
|---|---|---|---|---|
| baseline | **56.7** | 35.7 | 15.0 | 35.8 |
| reweight (adv-mag→softmax) | 54.4 | 34.6 | 15.0 | 34.7 |
| select (group-solve→band) | 52.9 | 35.2 | **5.0** | 31.0 |
| mix (reward-gap, 2-domain) | 55.5 | **36.1** | **17.5** | 36.5 |

This is **fully consistent with the literature**:
- **select hurt** (esp. AMC23 5.0) — plain GSM8K is uniformly easy (little difficulty
  spread) so DAPO-style filtering thins the batch; literature says filtering needs
  spread. Our band was also the naive (0,1); Difficulty Filtering's (0.2,0.8) is what
  actually helps.
- **mix best on generalization** — but only a 2-domain length split, where the
  literature predicts *marginal* gains; the small edge we saw is about what to expect.
- **our reweight ≠ the winning reweighter.** We used advantage-magnitude→softmax; the
  HIGH-evidence method is **Advantage Reweighting (low-prob token damping)**, a
  *different, token-level* rule we haven't implemented yet.

**Takeaway:** our current implementations are the "obvious" versions; the
literature's *winning* variants are specific (Difficulty Filtering's 0.2–0.8 band, AR's
token-level damping, learnability-signal mixing at ≥3 domains) and we haven't tried
them yet. That's the gap to close.

---

## Recommended next experiment (to actually beat baseline)

1. **Implement Advantage Reweighting** (token-level `w=α·π_θ+(1−α)`) — P0, cheapest HIGH-evidence win.
2. **Implement Online Difficulty Filtering** (reuse `group_solve_rate` scorer, band 0.2–0.8) — P0.
3. **Build a real ≥3-domain dataset** (math + code + logic/science, genuinely different difficulty) so mix/select have room to work — the 2-domain GSM8K split can't show mixture's value.
4. Re-run the 4-way comparison with (1)(2) and multi-domain data, **≥3 seeds**, more steps.

Deprioritize for small-model accuracy: GFPO, PF-PPO (no reward model), high-entropy
masking (large-scale-only), GRESO/PODS (efficiency not accuracy).

## Sources
Selection/Reweighting: arXiv 2503.14476, 2504.03380, 2508.09726, 2504.13818,
2506.02177, 2505.12929, 2409.06957, 2504.09696, 2506.01939.
Mixture: arXiv 2504.09710, 2506.08672, 1707.00183, 2312.02406, 2305.10429, 2509.19803;
MoDoMoDo / three-domain RLVR study / RECAP.
(Some exact deltas came from secondary summaries — arxiv direct fetch was blocked;
verify against PDFs before citing in a paper.)
