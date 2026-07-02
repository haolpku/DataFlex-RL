# DataFlex-verl 设计计划

> 把 DataFlex 的 **Data Select / Mix / Reweight** 能力扩展到 verl 的 RL(PPO/GRPO/…)场景。
> 目标交付形态:`pip install verl` + `pip install dataflex_verl`,改几行 YAML 即可启用,**不 fork verl**。

---

## 0. 结论摘要(TL;DR)

- **可行,且 verl 比 SFT 更适合做数据调度**——RL 信号更丰富(reward / advantage / log_prob / group),且 verl 自带三个**开放注册表**可挂载,无需改 verl 源码。
- **架构复用 DataFlex 顶层,重写 verl 适配层。** 复用 `Registry` / `components.yaml` / 配置体系;`Selector/Mixer/Weighter` 三个 HF-`Trainer` 子类**丢弃**(它们重写了 transformers 的 `_inner_training_loop`,与 verl 的 Ray + `fit()` 范式不兼容)。
- **核心抽象改造:`Scorer`(打分,共享)+ `Actuator`(执行,三分)。** 相同信号只打一次分;Selection/Reweighting/Mixture 因**挂载点、输出类型、成本语义**不同而必须拆开。
- **不按 PPO/GRPO 拆 selector。** 算法差异在 verl 里全封装在 advantage 注册表里,输出 batch 字段统一。selector 按**「消费的信号 × 粒度 × 时机」**分层,自然通吃 PPO/GRPO/未来算法。
- **打包为纯插件**,通过 verl 的三个注册表挂载:`TRAINER_REGISTRY`、`POLICY_LOSS_REGISTRY`、`ADV_ESTIMATOR_REGISTRY`(均已在源码核实)。

---

## 1. 背景:两个框架的关键差异

### 1.1 DataFlex 当前实现(基于 LLaMA-Factory)

- 入口 `dataflex-cli` → monkey-patch LLaMA-Factory(`cli.py`):替换 `FinetuningArguments`、把 `CustomSeq2SeqTrainer` 换成 `SelectTrainer/MixTrainer/WeightTrainer`。
- 三个 Trainer 继承 HF `Trainer`,**整段重写 `_inner_training_loop`**(`select_trainer.py` 900+ 行),在 warmup/update 节点插 `selector.select()`,并用 `DataLoader(Subset(dataset, indices))` 重建。
- 信号来源:SFT loss、梯度(LESS/NICE)、数据分布。粒度只有「一条样本」一层。

### 1.2 verl 训练范式(已核实,文件行号见附录 A)

- 入口 `main_ppo.py`(Hydra + Ray),`TaskRunnerV1.run()` 用 `get_trainer_cls(config.trainer.v1.trainer_mode)` 拿 trainer,调 `trainer.fit()`。
- `fit()`(`ray_trainer.py:1368`)手写循环,**不走 HF Trainer**:
  ```
  for batch in train_dataloader:            # StatefulDataLoader + RLHFDataset
      gen = rollout(batch)                   # 最贵的一步
      reward = extract_reward(batch)         # token_level_scores
      batch = compute_advantage(batch, adv_estimator=...)   # ← 算法唯一分歧点
      critic.update(batch); actor.update(batch)             # per-token policy loss
      if hasattr(train_dataset, "on_batch_end"): ...        # 现成钩子
  ```
- **PPO vs GRPO 的实质差异只在 `compute_advantage` 一处**,通过 `@register_adv_est` 分派(GAE/GRPO/RLOO/GDPO…),但**输出字段统一**写入 `batch.batch["advantages"]/["returns"]`,reward 统一在 `token_level_scores`。
- 粒度有三层:**prompt**(1 prompt → N rollouts)、**response**、**token**(`pg_losses` 形状 `(bs, resp_len)`)。

### 1.3 为什么不能直接搬代码

| | DataFlex(SFT) | verl(RL) |
|---|---|---|
| 训练循环 | HF `Trainer._inner_training_loop` | 手写 `fit()` + Ray worker group |
| 数据机制 | `Subset(dataset, indices)` 重建 loader | `StatefulDataLoader`,rollout 成本主导 |
| 模型前向 | 主进程 `model(**inputs)` | 分布式 worker group,不能直接 forward |
| 信号 | loss / grad | reward / advantage / log_prob / uid(group) |
| 样本粒度 | 1 层(样本) | 3 层(prompt/response/token) |

⇒ **顶层抽象与配置体系可复用;Trainer 层重写。**

---

## 2. 核心设计:Scorer(共享) + Actuator(三分)

### 2.1 为什么 Select/Reweight/Mix 要「打分合并、执行拆开」

数学上三者是同一个 per-sample 分数 `s_i` 的连续统一体:

| 机制 | 对样本 i | 权重语义 |
|---|---|---|
| Selection | `keep if s_i > τ` | 硬 0/1 |
| Reweighting | `w_i = g(s_i)` | 软 [0,∞) |
| Mixture | `p_d = agg({s_i : i∈d})` | 聚合到 domain 级 |

⇒ **打分函数 `f(signal)→s_i` 只写一次**(消除 DataFlex 里 `loss_selector` / `loss_weighter` 各自重算 loss + 分位数切分的重复)。

但执行层**必须拆开**,三个硬理由:

1. **挂载点不同(硬约束)**
   ```
   sampler ──► rollout ──► reward ──► advantage ──► policy loss
      ↑                      ↑                          ↑
   Mixture               Selection                 Reweighting
   (改采样概率)          (丢/留 prompt)             (乘 loss 权重)
   ```
   输出类型不同:`proportions` / `indices` / `weights`,无法用一个接口装。

2. **Selection 能省 rollout,Reweighting 不能(RL 特有,最关键)**
   pre-rollout 的 selection 直接跳过最贵的 generation;reweighting 必须已 rollout+算完 reward 才能乘权重。**「给权重 0」≠「选择性丢弃」**——在 rollout 成本主导的 RL 里,这是真金白银的算力差异(SFT 里没有此区别)。

3. **粒度天然绑定机制**:Mixture 只在 domain 级;Selection 在 prompt 级最自然;Reweighting 可下探 token 级。

### 2.2 类结构

```python
# ---- 打分层:共享 ----
class Scorer(ABC):
    requires: list[str]          # 依赖哪些 batch 字段 → 框架据此做兼容性校验
    timing: str                  # "pre_rollout" | "post_reward" | "post_advantage" | "in_loss"
    granularity: str             # "domain" | "prompt" | "response" | "token"
    needs_groups: bool = False   # True → 非 group 算法(PPO+GAE)拒绝运行

    @abstractmethod
    def score(self, batch, step_id, **ctx) -> Tensor:   # (bs,) 或 (bs, resp_len)
        ...

# ---- 执行层:三分,复用同一 Scorer ----
class Selector(Actuator):
    def act(self, scores, batch) -> list[int]:        ...   # indices,改 batch 成员
class Reweighter(Actuator):
    def act(self, scores, batch) -> Tensor:           ...   # weights,乘到 pg_losses
class Mixer(Actuator):
    def act(self, scores, batch) -> np.ndarray:       ...   # domain proportions
```

同一个 `reward_difficulty` scorer 可配成三种玩法,零重复打分代码:
- `selector: reward_difficulty + threshold`(DAPO 式过滤)
- `reweighter: reward_difficulty + softmax`(难度加权 loss)
- `mixer: reward_difficulty + group_agg`(难域配比)

### 2.3 为什么这能扛住「更新的算法」

新算法(DAPO/GSPO/Dr.GRPO…)在 verl 里 = 往 `ADV_ESTIMATOR_REGISTRY` 插一个新函数,只要遵守「输出标准 batch 字段」约定:
- 键于 **reward / advantage** 的 scorer → **自动继续可用,一行不改**。
- 只有当新算法引入了**你想据以选择的新信号**时,才加一个 scorer(不是改所有)。

⇒ scorer 数量 ≈ **选择准则种类**,而非 `算法数 × 准则数`。

---

## 3. 信号清单与「时机 × 机制」矩阵

selector 该按「读什么字段 + 何时可得 + 什么粒度」分层,而非按算法。

| Scorer 键于 | verl 字段 | timing | 可用 Actuator | 跨算法性 |
|---|---|---|---|---|
| 静态元数据/难度标注 | dataset 列 | pre_rollout | Mixer, **Selector(省 rollout)** | ✅ 与算法无关 |
| reward | `token_level_scores` | post_reward | Selector, Reweighter, Mixer | ✅ 通吃(advantage 之前就有) |
| advantage | `batch["advantages"]` | post_advantage | Selector, Reweighter | ⚠️ 字段通用,语义随算法不同(见下) |
| group 结构 | `non_tensor_batch["uid"]` | post_reward | Selector(过滤全对/全错组) | ⚠️ 仅 group 类算法(GRPO/RLOO/GDPO) |
| log_prob / KL | `log_prob`,`old_log_prob` | in_loss | Reweighter(token 级) | ✅ 通用 |

**语义提示(写进 config/文档,不需写两份代码):**
- PPO 的 advantage = critic-based per-token GAE;GRPO 的 = group 内归一化 outcome。同一段「选 |adv| 大的样本」代码两者都能跑,但**含义**不同。
- 键于 `uid` 的 scorer 设 `needs_groups=True`,框架在挂载时对 `adv_estimator` 校验,PPO 下直接报错/降级——**用一次配置检查替代 N 份按算法复制的代码**。

### 3.1 首批要实现的 Scorer(MVP)

| 名称 | 信号 | 说明 | 对标 |
|---|---|---|---|
| `reward_difficulty` | `token_level_scores` | outcome reward → 难度分 | DAPO dynamic sampling |
| `group_solve_rate` | `uid` + reward | group 内解出率,过滤全 0/全 1 组 | DAPO / RLOO 过滤 |
| `advantage_magnitude` | `advantages` | \|adv\| 大 = 学习信号强 | PF-PPO |
| `static_meta` | dataset 列 | 长度/来源/难度标注 | curriculum |

---

## 3.5 时序模型:SFT 离散相位 vs RL in-loop

一个常见误解:以为 RL 也是 DataFlex(SFT)那种「**选一批 → 训一段 → 再选**」的离散相位节奏。**不是。** 二者时序结构不同,且三个机制在 RL 里各自的时序也不同。

### 3.5.1 SFT(DataFlex 现状):离散相位,select 与 train 交替

`select_trainer.py` 的 `_inner_training_loop`:
```
warmup(选一批) → 训 warmup_step 步
              → select(全库打分,挑新子集,重建 dataloader)→ 训 update_step 步
              → select → 训 update_step 步 → ... 共 update_times 轮
```
select 是独立相位:暂停训练 → 用当前模型给**整个数据集**打分 → 重建 loader。

### 3.5.2 RL(默认):数据处理是每步循环内的一环,不是独立相位

verl `fit()` 每一步就是一条流水线,select/reweight 插在**同一 step 内**:
```
每个 step:
  取 batch → rollout(生成)→ reward → advantage → 更新
                              ↑
                    selection / reweighting 插这里(rollout 后、更新前)
```
原因:RL 的 reward/advantage 是**每批 rollout 出来才有**的,天然逐步产生。"选择"不是"从全库挑子集",而是"**对当前这批 rollout 出来的东西,决定谁进 loss、进多少权重**"(如 DAPO:一个 GRPO group 全对/全错 → advantage 全 0 → 就地丢弃)。

**只要 scorer 只读 batch 已有字段(reward/advantage),就每步都做,无需暂停。** 唯有需要额外全局 forward 的重型 scorer(如 LESS 式梯度打分)才回到 SFT 那种「训 N 步→暂停→全局打分」节奏,且在 verl 要走 Ray worker group(第 7 节风险)。

### 3.5.3 三种机制的信号时序对比

```
Mixture:      [domain 历史统计] ──滞后──► 调下一批的采样配比    (回顾性,per-domain,周期性)
Selection:    [本批 rollout 的 reward] ──即时──► 丢/留 prompt   (即时,per-prompt,每步)
Reweighting:  [本批的 advantage/reward] ──即时──► 乘 loss 权重  (即时,per-token/sample,每步)
```
三者信号**同源**(都是 reward/advantage),区别在三个维度:

| | 聚合粒度 | 时间性 | 作用对象 | 节奏 |
|---|---|---|---|---|
| Mixture | domain | 滞后累积 | 未来的采样分布 | 周期性(warmup+update,近似 SFT 相位) |
| Selection | prompt | 当场即时 | 当前 batch 成员 | 每步 in-loop |
| Reweighting | token/sample | 当场即时 | 当前 loss 贡献 | 每步 in-loop |

⇒ **只有 Mixer 真正保留 DataFlex 的「训一段→调→再训」离散节奏**;Selection/Reweighting 是每步内嵌。

### 3.5.4 Mixture 的信号从哪来 —— 回顾性、domain 级

Mixture 作用在 rollout **之前**,这一步的 reward 还没产生,所以它**不可能靠当前 batch 信号**——那是 selection/reweighting 的活。它靠的是「**过去若干步里各 domain 表现如何**」,滞后一步反馈给采样器:
```
step t:    按当前配比 p 抽样 → rollout → reward → 记录「domain d 这步表现 r_d」
step t+1:  继续累积各 domain 表现
每隔 K 步: mixer 读累积的 {domain → 表现} → 更新配比 p → 影响之后抽样
```
即:**用已算完的 reward/advantage,按 domain 聚合成滑动统计量,回填给采样器。** 信号与 selection 同源,消费方式不同(per-domain 滑动统计 vs per-sample 即时)。

**可用的 domain 级配比信号(RL 场景):**

| 配比信号 | 怎么算 | 直觉 / 对标 |
|---|---|---|
| domain 平均 reward / 解出率 | 按 uid group 聚合 reward,滑动平均 | 太易(全解出)/太难(全 0)都少抽,抽"正在学、有梯度"的 |
| domain 平均 \|advantage\| | advantage 绝对值滑动均值 | \|adv\| 大 = 还有学习信号 → 多抽(对标 ODM) |
| domain reward 提升速率 | `Δ(domain reward)/Δstep`,学习曲线斜率 | 进步快的多投入 → 课程学习 / 动态 DoReMi |
| domain 与 ref model 的 gap | policy vs ref 的 KL / reward gap | 差距大的优先(对标 DoReMi excess loss) |
| 静态先验 | 人工指定目标配比 | warmup 冷启动 fallback |

RL 里最自然的是前两个——reward/advantage 本就每步在算,按 domain 聚合几乎零额外成本(与 verl 现成的 `compute_pf_ppo_reweight_data` 同源,只是粒度从 sample 提到 domain)。

**冷启动(鸡生蛋):** 第一步还没任何 reward,配比无从算起 → Mixture **天然需要 warmup 相位**:先用静态先验(均匀/人工)跑 K 步纯攒 domain 数据,warmup 后 mixer 才有历史信号可读、开始动态调配比。这也是 Mixer 必须保留周期性节奏的第二个原因:单步 reward 噪声大,domain 配比不能一步一抖,须跨步累积;且改一次配比要生效若干步才能观察效果。

---

## 4. 三个执行机制在 verl 的挂载点(全部零 fork)

verl v1 提供三个**开放注册表**(已在 `core_algos.py` / `trainer_base.py` 核实):

1. **`register_trainer(name)`** → `TRAINER_REGISTRY`,由 `config.trainer.v1.trainer_mode` 选择。
2. **`register_policy_loss(name)`** → `POLICY_LOSS_REGISTRY`,由 `actor.policy_loss.loss_mode` 选择。
3. **`register_adv_est(name)`** → `ADV_ESTIMATOR_REGISTRY`,由 `algorithm.adv_estimator` 选择。
4. 补充:`load_class_from_fqn` 支持按全限定名动态加载(config 里填类路径即可)。

> **重要修正(已核实 v1 源码):verl 用的是 v1 `PPOTrainer` ABC,自带生命周期钩子,无需重写 `fit()`。**
> `verl/trainer/ppo/v1/trainer_base.py` 的 `PPOTrainer` 提供成对钩子:
> `on_train_begin/end`、`on_step_begin/end`、`on_sample_begin/end`(空实现,供子类 override)。
> 其 `step()`(`trainer_base.py:411`)是编号清晰的流水线:
> ①`_add_batch_to_generate` ②`replay_buffer.sample` ③`_compute_reward_colocate` ④`_balance_batch`
> ⑤`_compute_old_log_prob` ⑥ref ⑦values ⑧`_compute_advantage` ⑨`_update_critic` ⑩`_update_actor`。
> 三个内置 trainer(`sync`/`separate_async`/`colocate_async`)都继承它。
> ⇒ Selector/Mixer 挂载方式从「复制 `fit()` 循环体」降级为「**子类 override 一两个钩子 + override `_compute_advantage`/`step`**」,避免 DataFlex 在 SFT 侧那 900 行 `_inner_training_loop` 的重演。

### 4.1 Reweighter(最先做,改动最小)

- **挂载**:`@register_policy_loss("dataflex_weighted")` 注册一个包装 vanilla PPO loss 的函数,在聚合前把 `pg_losses (bs, resp_len)` 乘上 scorer 给的权重。verl 的 `compute_policy_loss_vanilla` 已有 `rollout_is_weights` 这个 per-token 乘权重口子,直接复用同一位置。
- **启用**:`actor_rollout_ref.actor.policy_loss.loss_mode: dataflex_weighted`。
- 语义最清晰、无需碰 `fit()` 主循环 → **MVP 第一步**。

### 4.2 Selector(rollout 后过滤)

- **挂载**:自定义 trainer 子类 `@register_trainer("dataflex_sync")`,继承内置 `sync` trainer,**override `_compute_advantage`(或在 `step()` 中 reward 之后、advantage 之前插一步)**,调 `selector.act(scores, batch)`,对 batch 按 index 取子集(丢弃 prompt 及其所有 rollout)。无需重写 `fit()`。
- 也可退化为**轻量方案**:override `on_sample_begin` 做 pre-rollout 过滤/采样,省 rollout。
- **启用**:`config.trainer.v1.trainer_mode: dataflex_sync`。

### 4.3 Mixer(多源配比)

- **挂载**:多源 prompt dataset + 自定义 sampler,override `on_step_end` 累积 domain 统计、每 K 步调 `mixer.act()` 更新 proportions,改采样概率。
- 落在 pre-rollout,与 DataFlex 的 `mixed_proportion_manager` 思路一致,但采样器换成 verl 的。

---

## 5. 打包方案:`pip install dataflex_verl`

### 5.1 设计原则:纯插件,零 fork

verl 的三个注册表 + FQN 加载 ⇒ `dataflex_verl` 只需在 import 时把自己的组件注册进去,**完全不改 verl 源码**。

### 5.2 安装即注册(entry point 机制)

```toml
# DataFlex_Verl/pyproject.toml
[project]
name = "dataflex_verl"
dependencies = ["pyyaml"]                  # verl 作为 host 框架,列为 optional 不强装

[project.optional-dependencies]
verl = ["verl"]

[project.entry-points."verl.plugins"]      # 若 verl 支持插件发现;否则见 5.3
dataflex = "dataflex_verl:register_all"
```

`register_all()` 调用 verl 的 `register_trainer` / `register_policy_loss` / `register_adv_est` 把组件挂上。

> **打包决策(已定):只发布 `dataflex_verl` 单个包,不和 `DataFlex`(SFT)同仓、不发独立 `dataflex-core`。**
> 因此框架无关的 Registry + Scorer/Actuator 基类作为 **`dataflex_verl.core` 内部模块**存在(不对外暴露成独立发行物)。
> 好处:`pip install dataflex_verl` 即全部,无额外依赖、无跨包版本对齐。抽象两层(Scorer/Actuator)的设计价值不变。

### 5.3 两条挂载路径(取其一,视 verl 版本)

- **路径 A(推荐,零改动)**:用户在训练脚本/config 顶部 `import dataflex_verl`(触发注册),或用 verl 的 `load_class_from_fqn` 在 YAML 里直接写 `dataflex_verl` 的类全名。
- **路径 B(自动发现)**:若 verl 支持 entry-point 插件扫描,则 `pip install` 后自动注册,连 import 都省。

### 5.4 用户视角(目标体验)

```bash
pip install verl
pip install dataflex_verl
```

```yaml
# 在标准 verl ppo_trainer.yaml 基础上,加几行:
trainer:
  v1:
    trainer_mode: dataflex          # 启用 DataFlex trainer(Selector/Mixer 挂载点)
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: dataflex_weighted  # 启用 Reweighter

dataflex:                           # DataFlex 专属配置块
  scorer: reward_difficulty
  actuator: selector                # selector | reweighter | mixer
  params:
    threshold: [0.05, 0.95]         # 过滤全错/全对
  warmup_step: 10
  update_step: 10
```

```bash
python -m verl.trainer.main_ppo --config-name ppo_trainer   # 原生命令,无需 dataflex-cli
```

### 5.5 仓库结构建议

```
DataFlex_Series/
├── DataFlex/                   # 现有 SFT 项目(LLaMA-Factory);不与 verl 版同仓发布
└── DataFlex_Verl/              # 唯一发布物,pip 包名 dataflex_verl
    ├── pyproject.toml
    ├── tests/                  # test_core.py(框架无关单测,12 passed)
    └── src/dataflex_verl/
        ├── __init__.py         # register_all() + 导出
        ├── core/               # 框架无关抽象层(内部模块,非独立包)
        │   ├── registry.py     # Registry + register_scorer/selector/reweighter/mixer
        │   ├── scorer.py       # Scorer ABC(requires/timing/granularity/needs_groups)
        │   ├── actuator.py     # Selector / Reweighter / Mixer ABC
        │   └── config.py       # load_component + validate_compat
        ├── policy_loss.py      # @register_policy_loss("dataflex_weighted")   (M1)
        ├── trainer.py          # @register_trainer("dataflex_sync")           (M2)
        ├── scorers/            # reward_difficulty / group_solve_rate / ...
        └── configs/components.yaml
```

> 关键:框架无关的 `Registry` 及 Scorer/Actuator 基类放在 `dataflex_verl.core` 内部模块。抽象两层(打分/执行)的解耦价值保留,但不作为独立包发布——`dataflex_verl` 是唯一发行物。

---

## 6. 实施路线图(里程碑)

| 阶段 | 交付 | 验收 |
|---|---|---|
| **M0 core 层** | `dataflex_verl.core`:Registry + Scorer/Actuator ABC + config | 单测:注册/构建/依赖校验通过 ✅ |
| **M1 Reweighter PoC** | `@register_policy_loss("dataflex_weighted")` + `reward_difficulty` scorer | GRPO 小模型跑通,loss 权重生效,指标可打印 |
| **M2 Selector** | `@register_trainer("dataflex")` + rollout 后过滤 | 复现 DAPO 式「过滤全对/全错组」,对比 baseline 收敛 |
| **M3 Mixer** | 多源 sampler + proportion 更新 | 两个 domain 配比随训练动态变化,可视化 |
| **M4 打包** | `pip install dataflex_verl` + entry-point 注册 | 干净环境安装,改 YAML 即用,无需改 verl |
| **M5 文档/示例** | `examples/verl/*.yaml` + skill 文档 | 对齐 DataFlex 现有 `how_to_add_algorithm.md` 风格 |

**建议起点:M1(Reweighter)** —— 不碰 `fit()` 主循环,只注册一个 policy loss,语义最清晰、验证最快。

---

## 7. 风险与开放问题

1. **verl 版本兼容**:注册表 API(`trainer_mode`、`v1`)属于较新的 verl v1,需锁定支持的 verl 版本范围;v0 路径需另做适配或不支持。
2. **分布式信号采集**:Selector/Scorer 若需额外 forward(如键于 loss/grad),不能像 SFT 直接 `model(**inputs)`,得走 Ray worker group ——**优先设计只消费 batch 里已有字段的 scorer**,避免额外前向。
3. **粒度一致性**:prompt 级过滤要连带丢弃该 prompt 的所有 rollout(GRPO group 完整性);token 级 reweight 要对齐 `response_mask`。
4. **entry-point 自动发现**:需确认目标 verl 版本是否支持插件扫描;不支持则退回「import 触发注册」路径 A。
5. **config 校验**:`dataflex` 配置块需接入 verl 的 `validate_config`,或独立校验,避免与 Hydra schema 冲突。

---

## 附录 A:verl 关键代码位置(已核实)

| 组件 | 位置 |
|---|---|
| `fit()` 主循环 | `verl/trainer/ppo/ray_trainer.py:1368` |
| dataloader 构建 | `ray_trainer.py:374-441`(`StatefulDataLoader` + `create_rl_dataset` + `create_rl_sampler`) |
| `compute_advantage`(算法分派) | `ray_trainer.py:187-282` |
| reward 提取 | `verl/trainer/ppo/reward.py:160`(`token_level_scores`) |
| per-token policy loss | `verl/trainer/ppo/core_algos.py:1278`(`compute_policy_loss_vanilla`,含 `rollout_is_weights` 口子) |
| loss 聚合 | `core_algos.py:1138`(`agg_loss`) |
| PF-PPO 重加权(现成参考) | `core_algos.py:2192`(`compute_pf_ppo_reweight_data`) |
| `TRAINER_REGISTRY` / `register_trainer` / `get_trainer_cls` | `verl/trainer/ppo/v1/trainer_base.py:1576-1602` |
| `POLICY_LOSS_REGISTRY` / `register_policy_loss` | `core_algos.py:50-85` |
| `ADV_ESTIMATOR_REGISTRY` / `register_adv_est` | `core_algos.py:113-129` |
| trainer 选择入口 | `main_ppo.py:134`(`get_trainer_cls(config.trainer.v1.trainer_mode)`) |
| 动态类加载 | `verl/utils/import_utils.py:208`(`load_class_from_fqn`) |
| 数据集后置钩子 | `ray_trainer.py:1770`(`train_dataset.on_batch_end`) |

## 附录 B:与 DataFlex(SFT)的接口映射

| DataFlex(SFT) | dataflex_verl(RL) |
|---|---|
| `Selector.select(model, step_id, num_samples) -> List[int]` | `Scorer.score(batch, step_id) -> Tensor` + `Selector.act(scores, batch) -> List[int]` |
| `Weighter.get_weighted_loss(losses, ...) -> scalar` | `Reweighter.act(scores, batch) -> Tensor`,挂 `register_policy_loss` |
| `Mixer.mix(model, step_id) -> proportions` | `Mixer.act(scores, batch) -> np.ndarray`,挂自定义 sampler |
| `Subset(dataset, indices)` 重建 loader | `DataProto` 取子集 / sampler 改概率 |
| loss / grad 信号 | reward / advantage / log_prob / uid |
| 1 层粒度 | domain / prompt / response / token |
