# 05 — 把 RL 与 OPD 融合进 DataFlex-RL:开发计划

> 目标:让 DataFlex-RL 在**纯 RL、纯 OPD、RL+OPD 混合**三种训练形态下都能做数据调度
> (select / reweight / mix),复用现有的 Scorer→Actuator 架构,**零改或最小改 verl**。

---

## 0. TL;DR(结论先行)

- **不新建 trainer。** verl 的 OPD 不是独立 trainer,而是 `PPOTrainerSync` 上的一组开关
  (`distillation.*`);DataFlex 的 `dataflex_sync` / `dataflex_mix_sync` 都继承自它,
  钩子点 `_compute_advantage` 在 OPD 下依然命中。
- **OPD 对 DataFlex 是"多了一路信号源"**:rollout 阶段 teacher 打分,产出 `teacher_logprobs`
  字段;到钩子点时它和 `old_log_probs` 一样已在 batch 里。新增**一个** `distill_kl` Scorer 即可,
  三个 Actuator(Selector/Reweighter/Mixer)全部复用。
- **梯度注入分两条路径**(已核对 verl 源码 `trainer/distillation/losses.py`):
  - **PG OPD**(`use_policy_gradient=true`)走 PPO policy-gradient,**显式传 `rollout_is_weights`**
    (`losses.py:281`)→ DataFlex 的 reweight/select **零改生效**。
  - **GKD OPD**(`use_policy_gradient=false`)走监督式直接 backprop,`agg_loss` **不乘** `rollout_is_weights`
    → reweight/select 当前**不生效**(缺口)。
  - **Mix 不经过 loss**(改采样分布),**任何模式都生效**。
- **锁定 PG OPD 作为一等公民**:它 == Thinking Machines "On-Policy Distillation",reverse-KL,
  三机制零改全生效。GKD 作为二等支持(仅 mix + 文档标注)。
- **最划算的组合 = RL+OPD 混合 loss + 双信号(reward + teacher 分歧)联合调度**:混合训练时
  teacher 本就对每个样本打分,蒸馏信号"顺手白拿",零额外成本。

---

## 1. 背景与关键事实(来自 verl 源码/文档核对)

### 1.1 verl OPD 架构(`docs/algo/opd.md`)
- Teacher logprob 在 **Agent Loop**(rollout 阶段)算,每样本 rollout 一完成即异步触发,
  打包成 `teacher_logprobs`(形状 `(S, 1 or K)`)塞进 batch。
- Student 优化跑在**同一批 actor workers**;actor 的 `loss_fn` 在 worker init 时从
  `ppo_loss` 切成 `distillation_ppo_loss`。
- 混合 loss:`use_task_rewards=true` → `L = L_policy + λ·L_distill`(`opd.md:452-476`)。
- 单/多 teacher:`teacher_key`(默认 `data_source`)路由 → 天然对接 DataFlex 的 domain/mix。

### 1.2 两种 OPD loss 变体
| | GKD OPD | PG OPD |
|---|---|---|
| 信号如何用 | 直接 backprop(监督式) | 当 reward 走 policy gradient(RL 式) |
| 散度 | forward KL,teacher **top-k 分布** | reverse KL,**单 token** 估计器(k1/k3) |
| teacher 返回 | top-k logits | 采样 token 的单个 logprob |
| verl 开关 | `use_policy_gradient=false`, `forward_kl_topk` | `use_policy_gradient=true`, `k1`/`k3` |
| 出处 | Agarwal 2024 | Thinking Machines 2025 |
| **DataFlex reweight/select** | ❌ 不乘 `rollout_is_weights` | ✅ 原样生效 |

### 1.3 梯度注入核对(`trainer/distillation/losses.py:230-296`)
```python
if loss_config.use_policy_gradient:           # PG OPD
    rollout_is_weights = data.get("rollout_is_weights", None)   # :273
    distillation_loss, _ = policy_loss_fn(
        advantages=-distillation_losses.detach(),
        rollout_is_weights=rollout_is_weights,                  # :281  ← DataFlex 权重生效
        ...)
else:                                          # GKD OPD
    distillation_loss = agg_loss(loss_mat=distillation_losses,
        loss_mask=response_mask, ...)          # 只有 mask,无 rollout_is_weights  ← 缺口
```

### 1.4 信号的正交性(为何对 RL 数据策略有用)
teacher 分歧 `k_t = student_logp − teacher_logp` 与 reward(对/错)**正交**:
- reward=1 但分歧大 → "蒙对了,过程烂",RL 会丢(adv≈0),OPD 信号救回;
- reward=0 但分歧小 → "teacher 也不会",RL 想训,OPD 信号提示别浪费。

---

## 2. 设计:新增信号,复用机制

### 2.1 新 Scorer(核心工作量)

**`distill_kl`**(per-token,granularity=`token`)
- `requires = ["old_log_probs", "teacher_logprobs", "response_mask"]`
- `timing = "post_advantage"`(和现有 token_prob 一致)
- 计算:`k_t = old_log_probs − teacher_logp`(reverse-KL 项;teacher_logprobs 取采样 token 那一列
  `[..., 0]`,与 verl k1/k3 估计器口径一致)。可选 `estimator ∈ {k1, k3, abs}` 复刻 verl 口径。
- 返回 `(bs, L)` per-token 分歧,供 token 级 reweighter 直接用。

**`distill_gap`**(per-sequence / per-domain 聚合,granularity=`prompt`)
- 同信号但按序列聚合 `D_i = Σ_t k_t / L_i`,供 selector / mixer 用。
- Mixer 场景再按 domain 求均值(复用 `DomainStatsTracker`)。

> 关键:**只加 Scorer,不加 Actuator。** 现有 selectors/reweighters/mixers 全部复用。

### 2.2 三机制映射(全部复用现有 Actuator)

| 机制 | Scorer | 复用的 Actuator | 语义 |
|---|---|---|---|
| **Reweight** | `distill_kl`(token) | `advantage_reweight` 或新 `per_token_power` | 分歧大的 token 加权,梯度聚焦分歧处 |
| **Select** | `distill_gap`(seq) | `topk_fraction` / `threshold_band` / `gfpo` | 组内保留分歧大的样本;丢 teacher-student 已一致的 |
| **Mix** | `distill_gap`(domain) | `tscl` / `dump_ucb` / 新 `distill_gap_mix` | 按各域平均分歧配比,teacher 领先多的域多采样 |

若现有 reweighter 语义不完全贴合,补一个 `distill_focus` reweighter:`w_t = (k_t)_+ ^α` 归一化
(只放大 student 落后 teacher 的 token),但**优先尝试复用 `advantage_reweight`/`per_advantage`**。

### 2.3 兼容性校验(挂载时,复用现有机制)
- Scorer 声明 `requires=["teacher_logprobs", ...]`;若 batch 无该字段(没开 teacher)→
  build 时明确报错:"distill_* scorer 需要 `distillation.enabled=true`"。
- 若 `mechanism ∈ {select, reweight}` 且检测到 `use_policy_gradient=false`(GKD)→
  **警告并拒绝/降级**:"GKD 路径不乘 rollout_is_weights,select/reweight 不生效,请用 PG OPD 或 mix"。
  (在 `build_from_config` 里读 `distillation.distillation_loss.use_policy_gradient`。)

---

## 3. 里程碑

### M1 — Scorer + 离线单测(不碰 GPU)
- [ ] `scorers.py` 加 `distill_kl` / `distill_gap`,读 `teacher_logprobs`(带 fallback 键名核对)。
- [ ] 单测:造 fake batch(含 `teacher_logprobs`),验证信号计算、聚合、mask 正确;
      验证无 teacher 字段时报错清晰。
- [ ] 三机制 dry-run 单测:distill_kl→reweighter,distill_gap→selector/mixer 各跑通。

### M2 — 兼容性校验 + config 打通
- [ ] `build_from_config` 读 `distillation.*`,做 §2.3 的 PG/GKD 校验与报错。
- [ ] 写 3 个 example config:`run_opd_reweight_pg.sh` / `run_opd_select_pg.sh` /
      `run_opd_mix.sh`(单 teacher,Qwen2.5 家族,PG OPD `k3`,`use_task_rewards` 可切)。
- [ ] 文档标注 GKD 限制。

### M3 — 端到端最小验证(排在现有 6 个 run 之后,GPU 空出再跑)
- [ ] **冒烟**:student==teacher,distill loss≈0(verl 官方推荐的自检,`opd.md:611`),
      验证 teacher 池起得来、`teacher_logprobs` 到达钩子点、DataFlex 信号非 NaN。
- [ ] **1 个真 run**:PG OPD + DataFlex reweight(distill_kl),300 步,确认
      `dataflex/weight_*` 生效、`actor/distillation/loss` 正常下降、无 OOM。
- [ ] teacher 资源池规划:7B student + 一个更大 Qwen2.5 teacher 的 GPU 切分
      (`n_gpus_per_node × nnodes` == teacher footprint,见 opd.md 约束)。

### M4 — 混合 loss + 双信号联合调度(目标场景)
- [ ] `use_task_rewards=true`(RL+OPD 混合),DataFlex 同时读 reward 与 teacher 分歧。
- [ ] 落地一个"只在分歧大的样本上开 OPD、其余走纯 RL"的 select 策略,验证省算力 + 不掉点。
- [ ] 统一 setting 对比:baseline(纯 GRPO)/ 纯 PG-OPD / RL+OPD 混合 / +DataFlex 调度。

### M5 — 文档 + 站点
- [ ] research/ 补一页 OPD 信号轴说明;doc site 加 `distill_kl` scorer 与 3 个机制页。
- [ ] README 增加 "RL + OPD" 章节与 quickstart。

---

## 4. 风险与开放问题

1. **GKD 缺口**:若日后要支持 GKD 的 reweight/select,需在 `agg_loss` 前乘 `rollout_is_weights`
   —— 这要**改 verl 源码**,违背零改原则。建议走上游 PR,不在 DataFlex 内 fork。
2. **teacher_logprobs 字段形状/键名**:文档说 `(S, 1 or K)`,需在真实 batch 里核对
   到达钩子点后的确切键名与 padding(nested tensor?),M1 单测用 fake、M3 用真 batch 双重验证。
3. **teacher 成本**:teacher 前向不便宜。Select 在 post-advantage 只省梯度不省 teacher 打分;
   要真省 teacher 算力得在 rollout 前的 replay-buffer 层 select(与 Mix 同层)——列为后续可选项。
4. **temperature**:teacher logprob 强制 `temperature=1.0`(opd.md:671),student rollout 若非 1.0,
   信号口径需注意(纯做调度信号影响小,但要在文档写清)。
5. **多 teacher(MOPD)× Mix**:`teacher_key=data_source` 与 DataFlex 的 `domain_key` 天然对齐,
   是一个很自然的"每域一个专家 teacher + DataFlex 按域配比"的组合,列为进阶实验。

---

## 5. 一句话路线

> 加一个 `distill_kl`/`distill_gap` Scorer,把 teacher 分歧接成 DataFlex 的第 4 路信号;
> 锁定 **PG OPD**(零改生效)+ 复用三个 Actuator;目标场景是 **RL+OPD 混合 loss 下用
> reward 与 teacher 分歧双信号联合调度**。不新建 trainer,不改 verl。
