# DataFlex-RL 4-way comparison — results

Controlled comparison: same model / data / optimizer / batch / seed; **only the
DataFlex mechanism differs**. Each run: Qwen2.5-0.5B, GSM8K (GRPO), 8× H20, **300
steps**. Checkpoints at `df_ckpts/<run>/global_step_300`.

Eval: official Qwen2.5-Math harness, `qwen25-math-cot`, temperature 0, full test
sets. `math` = the harness's MATH set (5000 problems), `amc23` = 40 problems.

## Scores (accuracy %)

| Run | GSM8K (1319) | MATH (5000) | AMC23 (40) | mean |
|---|---|---|---|---|
| baseline (stock verl GRPO) | **56.7** | 35.7 | 15.0 | 35.8 |
| reweight (adv-mag → softmax) | 54.4 | 34.6 | 15.0 | 34.7 |
| select (group-solve → band)  | 52.9 | 35.2 | 5.0 | 31.0 |
| **mix (reward-gap, 2-domain)** | 55.5 | **36.1** | **17.5** | **36.5** |

## Reading these numbers (important caveats)

- **Single seed, 300 steps, 0.5B.** Differences of 1-3 points are within run-to-run
  noise at this scale; treat this as a **pipeline / sanity comparison**, not a
  benchmark claim. No error bars — would need ≥3 seeds to be conclusive.
- **In-distribution (GSM8K):** baseline is nominally highest (56.7). The DataFlex
  mechanisms trade a little in-distribution fit; expected, since they reshape the
  gradient/data rather than purely maximize GSM8K reward.
- **Generalization (MATH / AMC23):** `mix` is best on both out-of-distribution math
  sets (36.1 / 17.5) and best on mean — consistent with the intuition that domain
  mixing (even the length-split gsm8k_short/long here) broadens rather than overfits.
- **select on AMC23 (5.0)** is the one clear outlier drop. With DAPO-style filtering
  on a single easy source (GSM8K), aggressive dropping of zero-signal groups can thin
  the effective batch and hurt the hardest transfer set; worth revisiting select's
  band on harder / multi-source data.

## What this validates
All four runs trained 300 steps cleanly on 8 GPUs and all four checkpoints eval end-
to-end through the shared harness — the training + merge + eval pipeline is sound and
the mechanisms are comparable head-to-head. For a publishable comparison: ≥3 seeds,
more steps, and multi-source training data (so mix/select operate on genuinely
distinct domains, not a length split of one source).

## Reproduce
```bash
# train the 4 runs (serial, 8 GPU each)
bash experiments/run_all.sh
# merge + eval step 300 on gsm8k/math/amc23
bash experiments/eval_all.sh
```
Raw generation logs: `df_logs/eval_<run>.log` (per-dataset `'acc'` lines).
