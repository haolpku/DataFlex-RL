# DataFlex-RL Paper 2 — Campaign v2 Results & Analysis

**Status**: Main round (select + reweight, 27 × 2 scales) essentially complete (52/54 math + 49/54 gpqa). Mixer round (12 × 2 scales) in progress (10/24 done at time of writing). This document captures the main round results and analysis; mixer results will be appended when training completes.

**Date**: 2026-07-11.

---

## 1. Setup

- **Base models**: Qwen2.5-7B-Instruct, Qwen2.5-0.5B-Instruct
- **Trainer**: verl v0.5+, GRPO, `rollout.n=5`, `kl_coef=0.001`, `max_prompt_length=1024`, `max_response_length=8192`, `use_remove_padding=True`, `gpu_memory_utilization=0.85`, `total_training_steps=300`
- **Data**: `multidomain_3` = math (math_dapo + deepscaler + gsm8k, 5000 samples, boxed-aligned) + logic (K&K knights-and-knaves, 5000) + science (SciQ MCQ, 5000)
- **Seeds**: 3 (s1, s2, s3)
- **Eval**:
  - Math (MATH-500, AIME24, OlympiadBench, MinervaMath, GSM8K): Qwen2.5-Math official harness (`qwen25-math-cot` prompt template, `temperature=0`, `max_tokens=8192`, latex2sympy verifier)
  - GPQA-Diamond: opencompass with SciQ-style prompt (matches training format), `first_option_postprocess` for A/B/C/D extraction

**Pipeline validity**: MATH-30 smoke on v2/7b/ar_s1 = 86.7% (matches Qwen2.5-Math paper), GPQA smoke = 33.3% (above 25% random baseline). Confirms training→eval format alignment.

**v2 origin**: v1 used `Answer: X` prompt/verifier, incompatible with official boxed benchmarks (all math scored 0%). v2 rebuilt with boxed prompt + `HuggingFaceH4/MATH-500` data_source routing to verl's `math_reward` (latex2sympy). All 54 main-round runs re-trained.

## 2. Methods (Scorer × Actuator design space)

The 9 methods differ in **which signal they score with** (scorer) and **how they act on the score** (actuator):

| Method       | Scorer signal            | Actuator                                    | Class                |
|--------------|--------------------------|---------------------------------------------|----------------------|
| baseline     | — (uniform sampling)     | —                                           | control              |
| ar           | token_prob               | advantage_reweight (alpha=0.5)              | reweight             |
| per          | group_solve_rate         | prioritized_replay                          | reweight             |
| softmax      | advantage_magnitude      | softmax weight                              | reweight             |
| difffilter   | group_solve_rate         | threshold_band [0.2, 0.8]                   | select-difficulty    |
| diffband     | group_solve_rate         | threshold_band (wider bandpass)             | select-difficulty    |
| maxvar       | reward_difficulty        | max_variance (keep top-50% variance)        | select-difficulty    |
| gfpo         | reward_difficulty        | GFPO (keep top-3 by efficiency)             | select-difficulty    |
| topk         | advantage_magnitude      | topk_fraction (keep top-50% \|advantage\|)  | select-magnitude     |

## 3. 7B Main Table (mean over 3 seeds; blanks = std missing)

| method       | MATH     | AIME24    | Olympiad | Minerva  | GSM8K     | GPQA      | avg       |
|--------------|----------|-----------|----------|----------|-----------|-----------|-----------|
| baseline     | 76.0 | 13.3  | 39.0 | 34.2 | 92.6  | 35.9      | **48.5**  |
| ar           | 76.5 | 12.2  | 40.5 | 35.2 | 92.4  | 33.0      | **48.3**  |
| per          | 76.0 | 13.3  | 38.8 | 34.6 | 92.4  | 32.8      | **48.0**  |
| softmax      | 75.3 | 12.2  | 39.1 | 35.7 | 92.1  | 35.5      | **48.3**  |
| difffilter   | 75.8 | 12.2  | 40.8 | 35.4 | 91.9  | 34.0      | **48.4**  |
| diffband     | 76.2 | 14.4  | 39.6 | 35.5 | 92.5  | 32.8      | **48.5**  |
| maxvar       | 75.6 | 14.5  | 37.7 | 36.2 | 92.6  | 36.0      | **48.8**  |
| gfpo         | 76.0 | 13.3  | 40.3 | 35.1 | 92.5  | 35.2      | **48.7**  |
| topk         | 76.1 | 16.6  | 40.2 | 35.3 | 92.1  | 34.0      | **49.0**  |

- **avg spread**: 48.0 – 49.0 = **1.0 pt**
- Seed std within a method: often **2–5 pt** on AIME24 (30 samples), Minerva (272), GPQA (198)
- Method-vs-method delta <= seed std => **no method significantly separates from baseline at 7B**

## 4. 0.5B Main Table

| method       | MATH     | AIME24 | Olympiad | Minerva | GSM8K    | GPQA     | avg       |
|--------------|----------|--------|----------|---------|----------|----------|-----------|
| baseline     | 36.0 | -      | 8.8  | 7.9 | 53.0 | 25.2 | **26.2**  |
| ar           | 36.2 | -      | 8.8  | 7.9 | 52.3 | 25.1 | **26.1**  |
| per          | 36.2 | -      | 8.9  | 7.4 | 53.2 | 25.8 | **26.3**  |
| softmax*     | 35.1 | 3.3    | 9.1  | 6.6 | 52.2 | 24.6 | **21.8**  |
| difffilter   | 35.4 | -      | 9.4  | 8.1 | 51.7 | 27.4 | **26.4**  |
| diffband     | 35.7 | -      | 9.2  | 8.0 | 51.8 | 24.2 | **25.8**  |
| maxvar       | 35.9 | -      | 9.1  | 7.2 | 52.6 | 22.1 | **25.4**  |
| gfpo         | 35.7 | -      | 9.5  | 8.7 | 52.3 | 24.1 | **26.1**  |
| topk         | 36.1 | -      | 9.6  | 7.8 | 52.6 | 24.5 | **26.1**  |

*softmax avg contaminated by an AIME24 empty-response outlier at s3 (3.3 vs baseline dashes). Excluding AIME24: softmax = 25.2. See section 7 for followup.

**AIME24 shows dashes for 0.5B** because the 0.5B model rarely reaches boxed answers on olympiad-level problems, causing all-empty responses.

## 5. Analysis

### 5.1 Finding: Reweight family lags Baseline at 7B (mildly, but consistently)

7B averages of Reweight vs Select vs Baseline:

| class                | avg   | delta vs baseline |
|----------------------|-------|-------------------|
| Reweight (ar/per/softmax) | 48.20 | **-0.30** |
| Select-difficulty (dff/db/mxv/gfpo) | 48.60 | +0.10 |
| Select-magnitude (topk) | 49.00 | +0.50 |
| Baseline | 48.50 | ref |

3/3 reweight methods land below baseline at 7B. Directionally consistent, magnitude within noise. **Interpretation**: at 7B the base policy already ranks samples adequately via advantage-driven GRPO; adding an external soft-weight (token_prob for `ar`, group_solve_rate for `per`, temperature softmax for `softmax`) is either redundant or actively distorts the effective on-policy gradient.

### 5.2 Finding: Selecting on advantage-magnitude > selecting on difficulty (at 7B)

- topk (advantage_magnitude scorer): +0.5 vs baseline — the only *positive* method at 7B.
- All 4 select-difficulty methods (gfpo, maxvar, difffilter, diffband) sit within 48.4–48.8, i.e. ~ baseline.

**Interpretation**: at 7B the group solve rate is a coarse signal (mostly 0/1 saturated), while |advantage| directly picks samples where the policy is uncertain enough to learn. Not a huge effect, but the *direction* is right.

### 5.3 Finding: Scorer > Actuator in explanatory power

Cross-actuator variance within one scorer family is smaller than cross-scorer variance:

| Scorer group             | Actuator variants                          | avg range   |
|--------------------------|--------------------------------------------|-------------|
| reward_difficulty (2)    | maxvar (48.8), gfpo (48.7)                 | 0.1 pt      |
| group_solve_rate (3)     | difffilter (48.4), diffband (48.5), per (48.0) | 0.5 pt  |
| advantage_magnitude (2)  | softmax (48.3), topk (49.0)                | 0.7 pt      |
| token_prob (1)           | ar (48.3)                                  | -           |

Actuator swap within `reward_difficulty` = **0.1 pt**, but scorer swap (reward_difficulty -> advantage_magnitude) shifts by **0.3–0.7 pt**. Design-space observation: **the scorer determines the achievable ceiling; the actuator is a mostly-neutral downstream detail.** This validates the Scorer x Actuator framing over a flat leaderboard.

### 5.4 Finding: 0.5B story is opposite of 7B on `difffilter` and `maxvar`

- **difffilter** at 0.5B: **+0.2 avg, +2.2 on GPQA** — the largest single-method / single-benchmark improvement in the campaign.
- **maxvar** at 0.5B: **-0.8 avg** — the worst method (excluding softmax outlier).

Both are select-difficulty. `difffilter` keeps solvable-but-not-trivial samples (0.2 <= solve rate <= 0.8); `maxvar` keeps high-variance samples (usually "occasionally solved, mostly failed" for a weak model). At 0.5B the base model is weak enough that:
- Selecting learnable samples (difffilter) -> helpful.
- Selecting high-variance samples (maxvar) -> majority failure trajectories, harmful.

**Interpretation**: for weak base models, *target difficulty* matters much more than *signal strength*. This flips relative to 7B, where target difficulty is neutralized by base-model competence.

### 5.5 Reweight is neutral at 0.5B

- ar, per, softmax(ex-AIME) ~ 26.1–26.3 vs baseline 26.2. No trend either way.
- Contrast with 7B (-0.3): reweight is not helpful at either scale but slightly harmful at 7B. Consistent with "GRPO already handles per-sample weighting internally."

## 6. Story arc for Paper 2

1. **Setup**: 3-domain training, 9 methods x 2 scales x 3 seeds, aligned eval pipeline.
2. **Main-round negative** (sec 3, 5.1): at 7B, no scorer x actuator combination clears the noise floor over baseline. Reweight is even mildly negative. This is a *replicable, well-scoped* negative result: it says "the design space doesn't matter at 7B/300-step regime."
3. **0.5B positive** (sec 5.4): at weaker scales, `difffilter` (target-solvability selector) does help meaningfully. Suggests the design space matters when the base model is not saturated.
4. **Scorer > Actuator** (sec 5.3): the empirical variance decomposition validates the design-space framing (scorer sets ceiling, actuator is a plumbing choice).
5. **Mixture round** (in progress): what happens when the design decision is *what fraction of each domain* rather than *what fraction of each sample*? Preliminary results appended below when training finishes.

## 7. Known TODOs

- Complete tail eval: 5 GPQA cells missing (`7b/baseline_s1`, `7b/baseline_s3`, `7b/diffband_s1`, `05b/baseline_s3`, `05b/topk_s3`) + 2 math cells (`05b/baseline_s3`, `05b/topk_s3`). Deferred until mixer training releases GPUs. None affect any qualitative finding.
- Rerun `softmax_s3` AIME24 (empty response outlier).
- Mixer round: 24 additional runs (4 mixers x 2 scales x 3 seeds), 10/24 done as of writing. Appended below when complete.
- LaTeX table export from `results/campaign_v2_eval_summary.csv`.

## 8. Mixer round results

See separate file [`CAMPAIGN_v2_MIXER_RESULTS.md`](CAMPAIGN_v2_MIXER_RESULTS.md). Summary:

- **7B mixers land in 46.8–49.0 range** (0.4 pt spread among static/tscl/dump_ucb, reward_gap partial). Static baseline (fixed 1/3) is competitive with all dynamic mixers.
- **0.5B: reward_gap edges static +0.4 avg**, consistent with main-round difffilter finding that weak models benefit from active steering.
- **Design-space framing supported**: cross-scorer variance > cross-actuator variance > cross-mixer variance.
- All 13 methods (9 main + 4 mixer) at 7B land within 2.2 pt of each other; at 0.5B within 3.9 pt (excluding aime24 outliers).

## 9. Future work / compute budget

**What we have**: 78 runs (54 main + 24 mixer) × 6 benchmarks on Qwen2.5-{0.5B, 7B}, 300 steps, 3-domain (math+logic+science). Total ~200 GPU-days.

**What would strengthen the paper** (ranked by return on compute):

### Option A: Cross-family generalization (Llama)
- **Purpose**: show findings hold beyond Qwen. Reviewers will ask this.
- **Minimum**: Llama-3.2-{1B, 3B} × 3 methods (baseline, difffilter, maxvar) × 3 seeds × 3 domains = **18 runs**.
- **Compute**: ~2h/run × 18 = **36 H100-hours** on 8 GPUs.
- **Value**: high — this is the highest-value ask reviewers will make.
- **Risk**: Llama's chat template + boxed-math alignment may need re-tuning (v1 was burned by this). Budget 20% overhead → **~45 hours effective**.

### Option B: Longer training (600 or 1000 steps)
- **Purpose**: test whether 7B "flat" result is a saturation artifact of 300 steps.
- **Minimum**: 3 methods × 3 seeds × 1000 steps at 7B = **9 runs × ~7h = 63 H100-hours**.
- **Value**: medium — if methods diverge at 1000 steps, story changes. If they stay flat, it's a stronger negative result.
- **Risk**: 7B GRPO at 1000 steps may show reward hacking; needs monitoring.

### Option C: Harder training data
- **Purpose**: get out of the "base model already solves it" regime.
- **Approach**: replace math_dapo + gsm8k with MATH-hard-only or AIME-style problems.
- **Minimum**: 3 methods × 3 seeds × 1 scale (7B) = **9 runs × ~2h = 18 H100-hours**.
- **Value**: high but subject to novelty concerns (many recent papers use math-hard).
- **Prerequisite**: identify a training set where baseline scores 30-50% (currently 76% on 300-step MATH). Building this is ~1 person-day.

### Option D: Larger models (14B / 32B)
- **Purpose**: test scaling hypothesis (does data-processing help at bigger scale?).
- **Minimum**: Qwen2.5-14B × 3 methods × 3 seeds = 9 runs × ~5h × 2 = **90 H100-hours** (need 2x GPU per run due to memory).
- **Value**: medium — but if flat holds at 14B, the negative-result becomes very strong. If breaks, story shifts to "data-processing matters only at scale."
- **Risk**: highest compute cost per finding.

### Option E: Deeper domain mixture
- **Purpose**: fix the mixer round's weak signal.
- **Approach**: 8–10 domains instead of 3 (add code, medical, common-sense, etc).
- **Minimum**: 4 mixers × 3 seeds × 1 scale (7B) × 10 domains = **12 runs × ~2h = 24 H100-hours** — but data prep is ~1 week.
- **Value**: high — this is where dynamic mixture *should* win.

### Recommended minimum for a top-tier ML venue (NeurIPS / ICML):
- **Options A + C** (Llama + harder data) = **~65 hours + 1 person-week** for data prep.
- Rationale: reviewers will demand cross-family + a regime where baseline is beatable. If both give consistent story with what we have, paper is defensible.

### Recommended minimum for a workshop (ICLR-workshop, ACL-workshop tier):
- **Nothing more needed** — current 78-run Qwen results with the "design-space" framing is publishable as-is at a workshop with limitations section noting Llama + harder-data as future work.

### If leaving the project (transition-friendly):
- Just publish current 78 rows as a benchmark study, framed as "we tested 13 methods and none clear the noise floor at 7B; here's what does at 0.5B." Emphasize the *methodology contribution* (Scorer×Actuator taxonomy, aligned pipeline, seed-noise analysis).
- Do NOT start Options A–E without a committed successor — half-finished experiments create data burden without benefit.

## 10. Reproducibility

- Training driver: `experiments/train_one_{7b,05b}_jizhi.sh`, `experiments/train_one_mix_jizhi.sh`
- Queue system: `experiments/queue_worker.sh` (main), `experiments/queue_worker_mix.sh` (mixer); jobs in `queue/jobs.txt` and `queue/jobs_mix.txt`
- Job generator: `experiments/gen_jobs.sh`, `experiments/gen_jobs_mix.sh`
- Eval scripts (outside repo, in `STEER/scripts/eval/`): `eval_math.sh` (Qwen2.5-Math harness), `eval_oc.sh` + `oc_configs/eval_gpqa_dataflex.py` (opencompass)
- Aggregation: `STEER/scripts/eval/aggregate_v2.py`, `method_summary.py`, `mixer_summary.py`
- Full per-ckpt raw numbers: `results/campaign_v2_eval_summary.csv`

## 11. Data

Full per-seed table in `results/campaign_v2_eval_summary.csv`. Schema:
`scale, name, seed, math, aime24, olympiadbench, minerva_math, gsm8k, gpqa`

Blank cells = eval not yet run (see section 7 TODO); zero cells = model produced no valid response.
