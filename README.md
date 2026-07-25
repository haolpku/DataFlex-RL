# DataFlex-RL

**DataFlex-RL** is a plug-in for [verl](https://github.com/volcengine/verl) that adds data-centric training controls to GRPO. It decomposes every method into two orthogonal choices — **what to score** and **what to do with the score** — and lets you compose them freely without modifying verl's core.

## Installation

```bash
pip install verl          # verl first
pip install dataflex-verl # then this package
```

## Quick start

Add two lines to your verl training command:

```bash
python -m verl.trainer.main_ppo \
    trainer.v1.trainer_mode=dataflex_sync \
    +dataflex.mechanism=select \
    +dataflex.scorer.name=group_solve_rate \
    +dataflex.scorer.params.success_threshold=0.5 \
    +dataflex.actuator.name=threshold_band \
    +dataflex.actuator.params.low=0.2 \
    +dataflex.actuator.params.high=0.8 \
    +dataflex.warmup_step=0 \
    ... # your normal verl args
```

That's it. DataFlex-RL hooks into verl's `_compute_advantage` step; no other changes needed.

## Scorer × Actuator abstraction

Every method is a combination of:

| Layer | What it does | Examples |
|---|---|---|
| **Scorer** | Assigns a value to each rollout or group | `group_solve_rate`, `advantage_magnitude`, `reward_difficulty`, `token_prob` |
| **Actuator** | Decides what to do with the score | `threshold_band` (select), `topk_fraction` (select), `advantage_reweight`, `softmax`, `per_advantage` (replay), `max_variance` (select) |

One scorer can feed any actuator, and one actuator can receive any scorer. A new method is a YAML config, not new code.

## Mechanisms

| `mechanism` | What changes | Notes |
|---|---|---|
| `select` | Which rollouts enter the policy update | Removes low/high-scoring groups before the loss |
| `reweight` | How much each rollout contributes | Multiplies policy loss by per-token weights |
| `mix` | Which domain is sampled next step | Dynamic curriculum over training domains |

## Built-in methods

```yaml
# Difficulty filtering (DAPO-style)
mechanism: select
scorer: group_solve_rate  success_threshold=0.5
actuator: threshold_band  low=0.2  high=0.8

# Top-k selection by advantage magnitude
mechanism: select
scorer: advantage_magnitude
actuator: topk_fraction  fraction=0.5

# Advantage reweighting
mechanism: reweight
scorer: token_prob
actuator: advantage_reweight  alpha=0.5

# Max-variance group selection
mechanism: select
scorer: reward_difficulty
actuator: max_variance  keep_fraction=0.5

# Domain curriculum (DUMP UCB)
mechanism: mix
scorer: reward_difficulty
actuator: dump_ucb  c=1.0  window=50
```

## Adding a new method

Subclass `Scorer` or `Selector`/`Reweighter`/`Mixer`, register with `@register_scorer` / `@register_selector` etc., and use it by name in the config. See `src/dataflex_verl/scorers.py` for examples.

```python
from dataflex_verl.core.registry import register_scorer
from dataflex_verl.core.scorer import Scorer

@register_scorer("my_scorer")
class MyScorer(Scorer):
    requires = ["old_log_probs", "response_mask"]
    timing = "post_advantage"
    granularity = "prompt"

    def score(self, batch, step_id, **ctx):
        # return a (batch_size,) tensor
        ...
```

## Tests

```bash
pip install -e ".[dev]"
pytest   # no GPU needed for unit tests
```

## Design

The scorer/actuator split and verl integration are described in `docs/DESIGN.md`. The key invariant: DataFlex-RL only writes `rollout_is_weights` into the TransferQueue; verl's policy loss picks it up automatically.
