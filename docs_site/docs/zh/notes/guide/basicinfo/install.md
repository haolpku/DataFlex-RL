---
title: 安装
icon: material-symbols:download
createTime: 2026/07/03 10:00:00
permalink: /zh/guide/basicinfo/install/
---

# 安装

## 安装

```bash
pip install verl          # 宿主 RL 框架(v1 trainer;默认 use_v1=true)
pip install dataflex_verl # 本插件
```

从源码(开发):

```bash
git clone https://github.com/haolpku/DataFlex-RL.git
cd DataFlex-RL
pip install -e ".[dev]"
```

## 冒烟检查(无需 GPU)

```bash
# 框架无关单测:scorer、actuator、注册表、兼容性校验
pytest -q

# 零配置自动注册:仅 import verl 就会注册我们的 trainer
python -c "
import verl
from verl.trainer.ppo.v1.trainer_base import TRAINER_REGISTRY
assert {'dataflex_sync', 'dataflex_mix_sync'} <= set(TRAINER_REGISTRY)
print('OK:', sorted(TRAINER_REGISTRY))
"
```

若两个 `dataflex_*` trainer 出现,说明 verl 的 entry-point 插件发现生效 —— 你自己不用 import
`dataflex_verl`,verl 的 Ray worker 也不用。

## 要求 / 兼容性

- **verl 的 v1 trainer**(`config.trainer.use_v1=true`,默认),使用 TransferQueue 数据面和
  `register_trainer` / 自定义采样器钩子。
- **Reweight / Select** 需要能填充标准 batch 字段的 advantage 估计器(GRPO、GAE、RLOO……)。
  分组 scorer(`needs_groups=True`)需要分组估计器(GRPO/RLOO/…)。
- 无需 flash-attn:示例用 `attn_implementation=sdpa` + `use_remove_padding=False`。

## 扩展提示

- **GPU:** 设 `CUDA_VISIBLE_DEVICES` 和 `trainer.n_gpus_per_node`。
- **Ray CPU:** 多核机器上,把 `ray_kwargs.ray_init.num_cpus` 设为 ~8×(GPU 数)。CPU 槽太少
  Ray 会在创建 colocated worker group 时死锁(现象:日志停在 `create worker group`,GPU 空闲)。
- **冷启动**(Ray + vLLM 引擎 + CUDA graph capture)在第 1 步前要几分钟,与 GPU 数无关,属正常。

## 排错

| 现象 | 原因 / 修复 |
|---|---|
| `Unknown trainer 'dataflex_sync'` | 插件未被发现 —— 重装;确认 `VERL_USE_EXTERNAL_PLUGINS` 不是 `none`。 |
| 停在 `create worker group`、GPU 空闲 | 调大 `ray_kwargs.ray_init.num_cpus`(~8×GPU 数)。 |
| `FlashAttention2 ... not installed` | 设 `+actor_rollout_ref.model.override_config.attn_implementation=sdpa` 且 `use_remove_padding=False`。 |
| `NotImplementedError: Reward function ... data_source=...`(mix) | 不要覆盖 `data_source`;域标签放单独列,设 `dataflex.domain_key`。 |
| `FileNotFoundError: Custom module file not found`(mix) | `custom_sampler.path` 要加 `pkg://` 前缀:`pkg://dataflex_verl.replay_buffer`。 |
