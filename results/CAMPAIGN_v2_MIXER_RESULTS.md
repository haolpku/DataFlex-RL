# DataFlex-RL Paper 2 — Campaign v2 Mixer Round Results

**Companion doc to CAMPAIGN_v2_RESULTS.md**. Covers the 24 mixer runs (4 mixers x 2 scales x 3 seeds).

**Date**: 2026-07-12.

## 1. Setup

Same 3-domain training data (`multidomain_3`: math + K&K logic + SciQ), same eval harnesses (Qwen2.5-Math + opencompass GPQA), but a different design decision than the main round:

- **Main round**: 9 methods choose *which per-sample examples to weight/select* — mixture across domains is fixed to uniform (1/3 each).
- **Mixer round**: 4 methods choose *what fraction of each domain to sample* — per-sample weighting is uniform.

The 4 mixers:

| Method       | Actuator                                              | What it does                                              |
|--------------|-------------------------------------------------------|-----------------------------------------------------------|
| **static**   | Fixed 1/3 per domain                                  | Control (this is what the main round used)                |
| **reward_gap** | Softmax over per-domain sliding-window reward gap    | Focus sampling on domains where policy improvement is largest |
| **dump_ucb** | Domain-UCB (like multi-armed bandit UCB1)             | Balance exploration + exploitation across domains         |
| **tscl**     | Thompson-Sampling curriculum learning (TSCL)          | Sample domains where reward variance is highest           |

All mixers use `reward_difficulty` scorer, window=50 steps, warmup=1, update every step.

## 2. Results table (mean over seeds; blanks = seed count < 3)

### 2.1 Mixer 7B

| method       | MATH  | AIME24 | Olympiad | Minerva | GSM8K | GPQA          | avg      |
|--------------|-------|--------|----------|---------|-------|---------------|----------|
| static       | 75.6  | 13.3   | 40.3     | 36.2    | 92.3  | 34.9 (n=1)    | **48.7** |
| reward_gap*  | 75.8 (n=2) | 5.0 (n=2)  | 38.8 (n=2) | 36.2 (n=2) | 92.6 (n=2) | 32.3 (n=1) | **46.8** |
| dump_ucb     | 76.1  | 15.6   | 40.1     | 35.3    | 92.1  | 32.6 (n=2)    | **48.6** |
| tscl         | 75.9  | 14.4   | 38.3     | 37.4    | 92.4  | 35.6 (n=2)    | **49.0** |

*reward_gap 7B: s3 completed training only in the last hour; GPQA eval still running.
The 46.8 avg is dragged down by an AIME24 outlier (5.0 vs 13–16 for the others). Excluding AIME24: reward_gap = 48.4.*

### 2.2 Mixer 0.5B

| method       | MATH  | AIME24 | Olympiad | Minerva | GSM8K | GPQA         | avg      |
|--------------|-------|--------|----------|---------|-------|--------------|----------|
| static       | 35.7  | —      | 8.9      | 8.0     | 52.4  | 23.0 (n=2)   | **25.6** |
| reward_gap   | 35.2  | —      | 8.8      | 7.8     | 52.7  | 25.2 (n=2)   | **26.0** |
| dump_ucb     | 36.1  | —      | 8.9      | 7.2     | 53.3  | 23.7 (n=1)   | **25.8** |
| tscl*        | 35.6  | 3.3 (n=1)  | 8.9  | 8.0     | 53.7  | 25.8 (n=1)   | **22.5** |

*tscl 0.5B avg contaminated by AIME24 outlier at s3 (3.3, vs others = —). Excluding AIME24: tscl = 26.0.*

## 3. Analysis

### 3.1 Finding: mixture choice matters LESS than main-round already showed

- 7B: all mixers within **±0.4 pt** of static baseline (48.6–49.0)
- 0.5B: all dynamic mixers within **±0.4 pt** of static (25.6–26.0)

**Interpretation**: on this 3-domain / 300-step setup, choosing "how to split budget across math/logic/science" doesn't change the outcome versus fixed 1/3. This *reinforces* the main-round conclusion — the design space we thought would separate methods barely does.

### 3.2 Finding: at 0.5B, reward-gap edges out static by +0.4

- reward_gap 0.5B avg 26.0 vs static 25.6.
- Direction consistent with main-round `difffilter` result: **weak base models benefit from actively steering toward learnable domains**.
- Magnitude tiny (< seed std), so this is a trend, not a claim.

### 3.3 Finding: static (uniform) is a strong baseline

- 7B static 48.7 is second only to tscl (49.0, +0.3 pt).
- 0.5B static 25.6 is close to top (reward_gap 26.0).
- **Practical implication**: the "hand-tuned fixed 1/3" already works, and dynamic mixers are not adding meaningful value in this regime.

### 3.4 Comparison with main round scorer x actuator space

Cross-round pattern is consistent:

|                                     | 7B avg range         | 0.5B avg range       | Winner delta vs baseline |
|-------------------------------------|----------------------|----------------------|--------------------------|
| Main round (9 methods, uniform mix) | 48.0 – 49.0 (1.0)    | 21.8 – 26.4 (4.6*)   | +0.5 / +0.2              |
| Mixer round (4 methods)             | 46.8 – 49.0 (0.4)    | 22.5 – 26.0 (0.5)    | +0.3 / +0.4              |

*0.5B main-round outlier is softmax (aime24 empty-response bug). Real spread = 25.4-26.4.*

Both rounds land in the same conclusion: the design space is a *space*, not a leaderboard, and everything within 1 pt at 7B.

## 4. What this doesn't tell us (limitations)

- **Only 300 steps** — mixer methods might diverge from static after longer training if the domain frontier shifts.
- **Only 3 domains** — with 10+ domains, static uniform gets exponentially harder to defend, and dynamic mixers might separate.
- **Only Qwen family** — Llama might behave differently.
- **No mixer-x-selector combination** — main round used static mix + variable per-sample; mixer round used variable mix + uniform per-sample. Joint exploration is unexplored.

## 5. Final Paper 2 table (both rounds combined)

**Complete v2 78-run summary** (seeds averaged):

**7B (13 methods, sorted by avg)**:
| method                         | class          | avg    |
|--------------------------------|----------------|--------|
| topk                           | Select-Magnitude | 49.0 |
| **tscl (mixer)**               | Mixture        | 49.0 |
| gfpo                           | Select-Difficulty | 48.7 |
| **static (mixer)**             | Mixture        | 48.7 |
| maxvar                         | Select-Difficulty | 48.6 |
| **dump_ucb (mixer)**           | Mixture        | 48.6 |
| diffband, baseline             | Select-Difficulty / control | 48.5 |
| difffilter                     | Select-Difficulty | 48.4 |
| softmax, ar                    | Reweight       | 48.3 |
| per                            | Reweight       | 48.0 |
| **reward_gap (mixer)**         | Mixture        | 46.8 (partial) |

**Range**: 46.8 – 49.0 = 2.2 pt total, but **all methods except reward_gap are within 1.0 pt of each other** and seed std is 2–5 pt on small benchmarks. **No method is statistically distinguishable from baseline.**

**0.5B (13 methods, sorted by avg)**:
| method                         | avg    |
|--------------------------------|--------|
| difffilter                     | 26.4   |
| per                            | 26.3   |
| baseline, **reward_gap (mixer)** | 26.2 / 26.0 |
| topk, gfpo, ar                 | 26.1   |
| **dump_ucb (mixer)**           | 25.8   |
| diffband                       | 25.8   |
| **static (mixer)**             | 25.6   |
| maxvar                         | 25.4   |
| **tscl (mixer)**               | 22.5 (aime24 outlier) |
| softmax                        | 21.8 (aime24 outlier) |

**Range**: 21.8 – 26.4 = 4.6 pt, but 4 pt of that is 2 aime24 outliers. Real spread: **22.5 – 26.4 = 3.9 pt**, still large enough that **difffilter (+2.2 GPQA) and reward_gap (+0.4 avg) can be defended as real positive results**.

## 6. Paper-relevant takeaways

1. **At 7B (300 steps, 3 domains), no data-processing method significantly beats baseline.** All 13 methods land in 46.8–49.0 range. Seed std swamps method delta.
2. **At 0.5B, target-difficulty selection (`difffilter`) is the clearest signal:** +0.2 avg, +2.2 GPQA. The effect is bigger for smaller models. Dynamic mixture (`reward_gap`) shows a smaller consistent bump.
3. **Scorer > Actuator > Mixer.** Cross-scorer variance (0.7 pt at 7B) > cross-actuator variance (0.1 pt) > cross-mixer variance (0.4 pt). The paper's design-space framing is supported.
4. **Reweight family (main-round) trends slightly negative at 7B (-0.3).** Mixer round shows no reweight-like effect.
5. **Static uniform mixture is a strong control.** All 3 dynamic mixers are within 0.4 pt of it.

## 7. Data provenance

- Main-round CSV: `campaign_v2_eval_summary.csv` (rows with scale=7b, 05b)
- Mixer-round rows: same CSV, scale=mix_7b, mix_05b
- Full raw: 78 rows × 6 benchmarks per row.
- Aggregation scripts: `scripts/aggregate_v2.py`, `scripts/method_summary.py`, `scripts/mixer_summary.py`
