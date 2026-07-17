# dataflex_verl

**DataFlex data-scheduling (Select · Mix · Reweight) for [verl](https://github.com/volcengine/verl) RL training — a zero-fork plugin.**

`dataflex_verl` brings DataFlex's data-centric training to RL. It plugs into verl's
open registries (`register_trainer`, custom replay-buffer sampler) without modifying
verl source. Install both, add a few YAML lines, run verl's normal entrypoint.

```bash
pip install verl
pip install dataflex_verl
```

## Design in one paragraph

Data scheduling splits into two layers:

- **Scorer** (`signal → score`): shared across mechanisms and RL algorithms. A scorer
  reads only the batch fields it declares (`requires`) — e.g. `advantages`,
  `token_level_scores`, group `uid` — so it works across PPO / GRPO / future
  estimators. Group-only scorers set `needs_groups=True` and are rejected at mount
  time on non-group algorithms.
- **Actuator** (`score → action`): three mechanisms that differ in *mount point*,
  *output type*, and *cost semantics*:

| Mechanism | Output | Mount point | When |
|---|---|---|---|
| **Reweight** | per-token weights → `rollout_is_weights` | trainer `_compute_advantage` | every step, in-loop |
| **Select** | 0/1 mask (drops gradient contribution) | trainer `_compute_advantage` | every step, in-loop |
| **Mix** | domain proportions → sampler | custom `ReplayBuffer` + trainer | periodic, pre-rollout |

Reweight and Select share one hook because both reduce to per-token weights that
verl's vanilla policy loss already multiplies in — **no custom policy loss needed**.
Mix is retrospective and per-domain: it accumulates each domain's mean reward and
steers *future* sampling, so it needs a warmup phase (cold start).

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full rationale, and
[`docs/QUICKSTART.md`](docs/QUICKSTART.md) for a copy-paste walkthrough.

## Usage

Enable via `config.dataflex` in verl's YAML (or `+dataflex.*` CLI overrides).

**Reweight** (emphasize high-|advantage| samples):
```yaml
trainer: {v1: {trainer_mode: dataflex_sync}}
dataflex:
  mechanism: reweight
  scorer:   {name: advantage_magnitude, params: {agg: mean}}
  actuator: {name: softmax, params: {temperature: 1.0}}
  warmup_step: 0
```

**Select** (DAPO-style: drop all-solved / all-failed GRPO groups):
```yaml
trainer: {v1: {trainer_mode: dataflex_sync}}
dataflex:
  mechanism: select
  scorer:   {name: group_solve_rate, params: {success_threshold: 0.5}}
  actuator: {name: threshold_band, params: {low: 0.0, high: 1.0}}
```

**Mix** (dynamic domain proportions; needs multi-source data):
```yaml
trainer:
  v1:
    trainer_mode: dataflex_mix_sync
    sampler:
      custom_sampler: {path: pkg://dataflex_verl.replay_buffer, name: DataFlexMixReplayBuffer}
dataflex:
  mechanism: mix
  domains: [gsm8k_short, gsm8k_long]   # names in the dataset's `domain` column
  domain_key: domain                   # column holding the domain label (default)
  scorer:   {name: reward_difficulty}
  actuator: {name: reward_gap, params: {temperature: 1.0, floor: 0.05}}
  warmup_step: 5
  update_step: 5
```

Mix needs a multi-domain dataset. Keep the real `data_source` column (verl uses it
to pick the reward function) and put the domain label in a separate column. Build a
demo split of GSM8K by question length:

```bash
python examples/build_2domain_gsm8k.py --src $DATA/gsm8k --dst $DATA/gsm8k_2domain
```

Runnable end-to-end scripts (Qwen2.5-0.5B / GSM8K / 8×GPU, verified) are in
[`examples/`](examples/): `run_reweight_grpo.sh`, `run_select_grpo.sh`, `run_mix_grpo.sh`.
On a many-CPU box, set `ray_kwargs.ray_init.num_cpus` to ~8×(#GPUs) or Ray can
deadlock creating the colocated worker groups.

## Registered components

| Kind | Names |
|---|---|
| scorer | `reward_difficulty`, `advantage_magnitude`, `group_solve_rate`, `token_prob` |
| reweighter | `softmax`, `difficulty_band`, `advantage_reweight`, `per_advantage` |
| selector | `threshold_band`, `topk_fraction`, `gfpo`, `max_variance` |
| mixer | `reward_gap`, `static`, `dump_ucb`, `tscl` |

## Adding a component

Subclass the relevant base and register it — see any file in `src/dataflex_verl/`.
A new scorer only needs `requires` / `timing` / `granularity` / `needs_groups` +
a `score()` method. Because scoring is shared, one scorer feeds all three mechanisms.

## Reproducing the paper evaluation

Every released checkpoint is evaluated on three domains (math, logic, science) at
`temperature=0`, `apply_chat_template`, seed 0, averaged over 3 training seeds.

**1. Merge** the FSDP checkpoint to HuggingFace format (required before eval):

```bash
python -m verl.model_merger merge --backend fsdp \
  --local_dir  <ckpt>/global_step_300/actor \
  --target_dir <ckpt>/global_step_300/actor/huggingface
```

**2. Evaluate.** Math uses the official Qwen2.5-Math harness (latex2sympy grader,
`qwen25-math-cot`, `max_tokens=8192`): MATH-500, AIME24, OlympiadBench, Minerva, GSM8K.
Logic and science are verl-schema parquets under `benchmarks/{logic,science}/`, scored
by `dataflex_verl.rewards.multidomain_reward.compute_score` (K&K exact-match; MCQ
letter-match), `max_tokens=4096`:

```bash
# logic (kk_hard, bbh_logical_deduction, bbh_tracking, zebra_logic_mc)
# science (mmlu_pro_chemistry, mmlu_pro_physics, gpqa_diamond)
python experiments/eval_benchmark.py \
  --model <ckpt>/.../huggingface \
  --benchmark benchmarks/logic/kk_hard.parquet \
  --out results/<exp>/kk_hard.json
# or amortize model load across all benchmarks:
bash experiments/eval_benchmarks_all.sh          # 8 GPUs = 8× TP=1 data-parallel
```

> **Parallelism note.** Use 8× TP=1 (one checkpoint per GPU), *not* tensor-parallel=8:
> Qwen-7B has 28 attention heads (legal TP ∈ {1,2,4,7,14,28}). For TP>1, set
> `NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_CUMEM_HOST_ENABLE=0`.

**3. Aggregate** per-seed benchmark scores into the domain tables (mean over seeds per
benchmark, then over benchmarks per domain):

```bash
python experiments/aggregate_benchmarks.py     # -> results/*.csv
```

GPQA-Diamond is reported as a secondary "hard ceiling" metric only: at ≤7B it sits near
the 4-choice random baseline, so MMLU-Pro is the primary science discriminator.

## Testing

```bash
pip install -e ".[dev]"
pytest            # framework-agnostic unit tests (no verl / GPU needed)
```

## Requirements / compatibility

- verl with the **v1 trainer** (`config.trainer.use_v1=true`, the default), which uses
  the TransferQueue data plane and the `register_trainer` / custom-sampler hooks.
- Reweight/Select need an advantage estimator that populates standard batch fields
  (GRPO, GAE, RLOO, …). Group scorers require a group-based estimator.
