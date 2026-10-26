# 50/100 车 MASAC 开发树 v3（2026-10-24）

本目录是 v2 返回包的**独立开发副本**，用于 50/100 车 MASAC 改造。原包
`返回_交接_同学_50_100车开发_v2_20261023(1)/` 一字未动，其 112 文件 sha256 清单仍然有效。

> **2026-10-25 目标函数口径**：目标是**每条链路各自的 `r_k/p_k` 再取平均**
> （`mean_pair_ee`）。`Σr/Σp`（network EE）与 `min_k(r_k/p_k)` 只作诊断输出，
> 不参与奖励、预算或选择。奖励、预算、参考动作、选择器、审计已全部按此口径贯通，
> 详见 `../../逐链路EE口径贯通_20261025.md`。本文下面若干历史数字来自旧口径，
> 已就地标注。

## 与 v2 的差异

1. **features v2**（`p3learn/features.py`）：`NODE_DIM` 8→12，
   `FEATURE_VERSION = "causal_normalized_12x12_v2"`。新增的 4 列是节点级最弱边摘要
   （关联合法 RF/VLC 候选的最小 log 增益、RF 候选平均相对方差、缺测或旧 CSI 比例），
   全部由同一个 `edge` 张量聚合，不引入第二套真相。空集哨兵写在模块文档里。
2. **`hetero_graph_v2`**（`p3learn/models.py`）：4 种有向关系 =
   {RF, VLC} × {合法候选边, 非候选但仍可干扰的边}，每种关系同时做均值与**最小值**池化，
   出/入两方向。干扰关系由块初掩码与信念增益导出（30 dB 窗口），不看已执行动作，
   因此不与解码器循环依赖。旧的 `mlp` / `hetero_graph` 行为完全不变。
   100 车默认走 `hetero_graph_v2`，50 车仍是 `mlp`。
3. **`p3learn/credit.py`**（新）：合法替换的反事实信用，**只做诊断**。
   四类替换（模式、方向、伙伴、功率）都必须是完整配对并通过块初候选/业务契约；
   VLC 端口一律取块初主端口表（actor 采样的 Action 在非 VLC 位置上端口为 0）。
   结果写独立的 `credit.json`，不进 `results.csv` / `summary.json` / NPZ，
   因为 `audit_allocation_outputs.py` 对方法集合、分母与数组名是严格匹配的。

## 运行

```powershell
$py = "D:\anaconda\envs\pymarl\python.exe"   # torch 2.0.0+cpu / Python 3.10

# 单测
& $py -B -I -m pytest -q -p no:cacheprovider tests

# 50 车 2 块训练 + 1 块评估（development 场景，不需要 manifest）
& $py -B -I run_allocation.py --n 50 --delta 0.05 --phase train --split development `
  --blocks 2 --start-block 2 --draws 4 --out ../runs/n50_train
& $py -B -I run_allocation.py --n 50 --delta 0.05 --phase evaluate --split development `
  --blocks 1 --start-block 4 --draws 4 --checkpoint ../runs/n50_train/model.pt `
  --credit --out ../runs/n50_eval

# 100 车：显式选旧编码器做对照，再跑 v2
& $py -B -I run_allocation.py --n 100 ... --encoder hetero_graph   --out ../runs/n100_hg_train
& $py -B -I run_allocation.py --n 100 ... --encoder hetero_graph_v2 --out ../runs/n100_hg2_train
```

`--credit-max-evals 0` 表示不截断（n=100 全量约 1400 个候选取值，明显变慢）；
默认 48，按已执行动作的最弱 3 条链路优先。

## 2026-10-24 修正（v3.1）

两处会妨碍学习的实现问题，按 agent1 复核意见修复：

1. **功率头改为对数均匀映射**：`p = 1 mW · (2000)^sigmoid(z)`，潜变量不再截断在 [-5, 5]。
   旧的 `1 mW + 1.999·sigmoid(z)` 在确定性评估下最低只能到 14.38 mW，而参考动作是 1 mW。
   现在 `z=-12` 给出 1.00005 mW，确定性策略可以表达参考动作的功率。
   雅可比同步改为 `log p + log ln 2000 + log sigmoid(z)(1-sigmoid(z))`。
2. **回放改为"提议标签"**：`--training-labels proposal`（默认）。
   回退发生时，仍然存 `(状态, 提议动作, 提议自己的预测平均逐链路 EE, 提议自己的预算缺口)`，
   不再把参考动作的好奖励贴给被拒提议。执行动作照旧写进 `results.csv` 用于评估。
   `--training-labels executed` 保留旧的"只学执行动作"行为，仅用于对照。
   `ReplayRecord.label_source` 记录每条样本的来源。

这两项使 checkpoint 版本升到 2：旧的 v2 包 checkpoint 与本次修复前的短跑 checkpoint 都不能再加载，必须重训。
单次 50 车 1 块试运行（`runs/trial_labels_proposal`）确认标签路径生效：
`label_source=proposal_forecast`、`predicted_budget_violation=0.947`、`mean_budget_cost=0.947`，
critic/actor loss 有限；这不证明能学，只证明反馈链条通了。

## 2026-10-25 修正（v3.1 续）：critic 看不到 1 mW 附近的功率

`CentralCritic` 原来把功率按 `p / 2` 送入：1 mW 与 1.05 mW 分别变成 0.0005 与 0.000525，
在 MLP 里不可分辨。固定 reference 配对、只扫功率时，Q 从 1 mW 到 100 mW 几乎不动
（0.9568 → 0.8935），而真实目标从 +0.998 掉到 −0.930——critic 对功率是"瞎"的，
actor 因此拿不到"往下压功率"的梯度。

改为 `critic_power_feature(p) = log10(p) + 3`（1 mW 记 0，2 W 记 3.301）。修复后同一扫描
Q 单调下降（1.0103 @1 mW → 0.9875 @1.05 mW → 0.6957 @2 W → −0.5982 @100 mW）。
这次改动使 checkpoint 版本升到 3：v2 及更早的 checkpoint 必须重训。

注意方向仍然偏浅：2 W 处 Q 仍比目标高约 0.65，说明功率项在 critic 里被系统性低估，
这不影响"能分辨"，但影响 actor 压功率的力度。

## 行为克隆 + 持续反馈探针

`probe_bc_feedback.py` 回答两个问题：策略**自己**生成动作能否接近参考性能；
在同样的初始化上继续 MASAC、并把每条新提议重新预测后写入回放，是否会优于仅克隆。

两个 arm 共用同一 `ModelConfig` 与 `SACConfig`：

1. `bc_only`：只用 reference 动作做监督（`imitation_loss`），配对/方向/模式按采样器
   自己的 anchor 顺序与同一合法掩码做交叉熵，功率在**对数潜变量**上做 MSE
   （`latent_from_power` 带 0.1% 端点容差，不对 1 mW 做无穷逆映射）。
   预训练结束后只让策略自己生成完整动作来评估。
2. `bc_masac`：从**同一**克隆后的 actor 出发，回放先填平衡集（reference、预算可行且
   提高平均逐链路 EE 的合法邻居、随机策略 coverage 动作、克隆策略提议），先只训 critic
   （`update(actor=False)`），在留出动作上达到 `MAE ≤ 0.2` 且 `concordance ≥ 0.75`
   才打开 actor 门。之后每轮 8 条新提议 → 冻结预测 bank 算它自己的平均逐链路 EE 与预算缺口 →
   入回放；每轮 3 次 critic 更新 + 1 次带 actor 的更新；reference 与改进邻居每轮重新加入，
   避免回放被一堆同样差的提议淹没。

```powershell
& $py -B -I probe_bc_feedback.py --n 50 --block 2 `
  --out ../runs/probe_bc_decomp_n50_b2 --save-checkpoints
```

输出还包含：`neighbours`（reference 的合法一步邻居里有多少真的提高平均逐链路 EE，即这个状态
还有多少空间）、`critic_warmup`、每 15 轮的 `heldout_critic`（未参与拟合的留出动作上的
MAE/concordance）、`gradient_split`（`-Q` 项、连续项、以及评分函数系数
`α_c·log π_c / −Q / α_d·(log π_d+1) / baseline` 四项各自的水平值与梯度范数）、
`power_response`（固定配对、扫功率时 Q 与目标值），以及 `floor_power_control`
（同一配对换成 reference 的 1 mW 功率），用来把"结构退化"与"功率上漂"分开。

注意评分函数系数是把四项**相加**后乘 `∂log π_d` 的，单看某一项的梯度会误判：
在 1 mW 附近 `α_c·log π_c ≈ 0.34`（被 `−log|dp/dz|` 抬高），是优势项的十几倍，
而离散熵本身只有 `~1e-3`。

`exploration` 块量的是"策略离那条好动作有多远"：对 4 条高价值邻居做教师强制，
给出 `log π_d` 与 reference 的差，以及**两条路径第一次分岔的那一步**上
`π(候选链路)` 与 `π(reference 链路)` 各自的概率——那是采样器真要产出这条动作
所需的样本量，不是"critic 认不认可"。同时统计 480 条提议与 reference、
与每条邻居的链路对称差直方图、最好的提议离邻居几条链路，以及 critic 在
**训练提议**上的 MAE/concordance（与留出集分开报）。

`selection` 块是一条**推理期**规则（不改训练）：`--selection-k 16` 采 16 条提议，
用同一批候选、同一个冻结 bank、同一个预算盾，比较**五个**选择器：
`critic`（`argmax Q`）、`training_label`（`argmax mean_ee/reward_scale − cost`，
即 critic 真正拟合的目标）、`feasible_forecast`（先筛 `cost == 0` 再取最小 EE 最大，
**与研究问题一致的那一条**）、`forecast_ignoring_budget`（`argmax mean_ee`，旧口径，
仅作对照）、`first_sample`（不挑）。重复 `--selection-repeats` 次。

> **2026-10-25 更正**：旧的 `by_forecast_oracle` 直接在**全部**候选上取
> `argmax(min_ee)`（当时的低标准），没有先筛预算可行性，而 critic 学的是
> `min_ee/reward_scale − cost`。两者优化的量不同，所以旧的
> `critic_equals_oracle_rate`（BC 1/4、MASAC 0/4）**不能当作排序错误的证据**，
> 该键已删除。现在同时报 `agreements`（三对一致率）、
> `concordance`（Q 与自身训练标签的成对 concordance，全体／仅可行子集）、
> `regret_predicted`（相对可行最优的预测 regret，旧口径允许为负）、
> `truth_*`（可选 `--truth`，决策全部固定后读取）以及
> `accounting`（bank 评估次数、耗时）。

实测（n=50，block 2，k=16，8 次重复，已有检查点，见
`../runs/selection_audit_n50_b2/audit.json`）：

| | BC | MASAC |
|---|---|---|
| 可行最优的预测 min EE（旧口径） | 9.970e8 | 9.788e8 |
| critic 选的预测 min EE（旧口径） | 7.329e8 | 8.617e8 |
| 首条样本的预测 min EE（旧口径） | 9.968e8 | 8.337e8 |
| Q 与训练标签 concordance（全体） | 0.547 | 0.684 |
| 同上，仅预算合格子集 | 0.544 | 0.555 |
| 每次重复的 bank 评估 / 耗时 | 26 / 1.32 s | 26 / 1.21 s |

结论：critic **能分开可行与不可行**（MASAC 的 critic 选 0% 需要回退，旧口径 38% 需要），
但**排不了合格候选之间的细差异**（≈随机）。门控里报的 0.855 来自"坏提议 + 克隆好提议"
的双峰留出集，只说明能分开好坏两类。BC 的首条样本已接近可行最优（差 0.02%），
所以克隆阶段"多采几条再挑"没有收益；MASAC 的分布里确实有好动作但 critic 抓不到。

`probe_selection_audit.py` 用**已有检查点**跑上面这张表（不训练）：

```powershell
& $py -B -I probe_selection_audit.py --n 50 --block 2 --k 16 --repeats 8 `
  --checkpoint bc_only  ../runs/probe_bc_decomp_n50_b2/actor_bc.pt `
  --checkpoint bc_masac ../runs/probe_bc_decomp_n50_b2/actor_masac.pt `
  --out ../runs/selection_audit_n50_b2
```

`probe_bc_feedback.py` 另有 `--branch original|no_continuous_entropy|fixed_power`
与 `--truth`。`fixed_power` 会把 actor 头最后一层的第 0、2 行（功率潜变量均值、
`log_std`）每次优化步后清零并置 bias −12 / −5，功率恒定在 1 mW，离散通道不动。
**它只排除功率上漂，不改变离散采样概率**，是诊断分支，不是解决方案。
`probe_branch_scan.py` 把三分支 × 多种子跑成独立目录并聚合：

```powershell
& $py -B -I probe_branch_scan.py --n 50 --block 2 --rounds 60 `
  --seeds 8 9 10 --jobs 3 --truth --save-checkpoints --out ../runs/branch_scan_n50_b2
```

另注意 MASAC 提议的小样本波动很大：4/8/16 条样本的均值 7.05e8/9.51e8/（最好 1.31e9）。
三组来自不同样本集合，不能合起来证明分布重尾，只说明单状态下小样本均值只能看量级。

`analyze_bc_checkpoints.py` 读取上面存下的 checkpoint，做两件离线诊断：把确定性动作的
配对与 reference 逐条对比，以及直接对功率潜变量求 `∂Q/∂z` 与 `∂(α_c·log π_c)/∂z`
（actor 最小化 `α_c·log π_c − Q`，所以看 `−∂Q/∂z + ∂(α_c·log π_c)/∂z` 的符号），
看目标到底把功率往哪推、哪一项占优。实测（n=50，block 2）：BC 刚训完时
`∂Q/∂z ≈ +4e-8`、`∂(α_c·log π_c)/∂z ≈ −1.0e-3`，熵项比收益项大约四个数量级；
MASAC 60 轮后两者为 `−6.4e-4` 与 `−1.0e-3`，熵项仍略占优，更新继续把功率往上推。
原因是 `log|dp/dz|` 里的 `log σ(z)(1−σ(z))` 在地板附近斜率≈1，而 `dp/dz≈1e-6` W/单位 z，
使 Q 对潜变量的斜率随功率一起缩小。

```powershell
& $py -B -I analyze_bc_checkpoints.py `
  --checkpoint bc ../runs/probe_bc_decomp_n50_b2/actor_bc.pt `
  --checkpoint masac ../runs/probe_bc_decomp_n50_b2/actor_masac.pt
```

## 固定状态学习探针

`probe_fixed_state.py` 在一个真实块、一个冻结状态上，用 reference 动作与若干条
策略提议各自**自己的**预测做标签，然后只在这个固定回放上训练，最后比较
critic 的排序与 actor 提议的变化。它回答的是"学习链条有没有梯度方向"，
**不是**训练收敛，也不能拿来排算法。

```powershell
& $py -B -I probe_fixed_state.py --n 50 --block 2 --updates 300 `
  --samples 4 --out ../runs/probe_fixed_state_n50_b2
```

`--updates` 是固定状态上的更新次数；`--samples` 是初始提议条数。
输出 `probe.json`，含 `before`/`after` 的 critic 排序、确定性提议的预测平均逐链路 EE 与预算缺口、
以及 `protect_budget` 下的回退率。

## 边界

- `credit.json` 是条件轨迹**预测**口径，不是真值收益；一次替换会改变所有链路的 SINR，
  没有逐链路 EE 归因，只有动作级 Δ 与逐槽位速率增量。
- 2 块短跑不能证明收敛，也不能用来排算法优劣。回退率仍是首要观察对象。
- `FEATURE_VERSION` 变更使 v2 的所有 checkpoint 失效；旧 checkpoint 不能用本树评估。
- 关系划分（30 dB 窗口、VLC 端口代理）是建模选择，没有真值；VLC 干扰关系只覆盖
  三到六成节点，很多节点拿到零摘要，这是信息限制而不是缺陷。
- 没有恢复 QoS 硬约束，没有改 γ，没有加业务队列。


## 2026-10-26 10车连续功率与1Mbit/s约束修复

本节覆盖此前10车默认八档网格、无优化QoS的描述。50/100车默认行为未调整。

- 目标仍是每条链路 `R/(P_circuit+p)` 后取算术平均；RF固定项0.1159W、VLC1.745W不变。
- 10车入口默认 `--power-solver continuous --q-min-bps 1000000`。全块100ms固定动作，无块内轮换或跨块补偿。
- `p3learn/continuous_power.py` 使用分式和二次变换与显式约束SCA。功率在[0.001,2]W连续区间求解；八档只初始化。SLSQP是连续凸子问题的数值求解器，不是算法名称，也不代表整体非凸问题全局最优。
- 内层基于块初信念增益代理，最终用同一个条件轨迹评估器筛选；预测块平均速率每条至少1Mbit/s才可返回成功。真实执行另查1Mbit/s，预测达标不保证真实达标。
- 正QoS未找到可行候选时返回 `qos_not_found`，不宣称物理上不可能。当前带约束开发仅支持 `--methods reference`，通过 `--reference-search candidates/joint` 分别运行同约束基线和联合方法；不允许旧学习或maxmin路径悄悄绕过QoS。
- `continuous_power_audit.json`记录每个候选连续子问题状态、迭代次数、残差及是否接受。数值失败或达到迭代上限不能包装成收敛。
- 无QoS修复诊断需显式 `--q-min-bps 0`；旧网格复现需再加 `--power-solver ascent`。这两项不是当前主配置。
- `--power-solver dinkelbach`保留旧有截断的历史分支，不作为修复后的SCA主算法。

两组公平比较命令（其余随机种子必须相同，结果目录不能复用）：

```text
python run_allocation.py --n 10 --delta 0.1 --methods reference --blocks 10 --start-block 2 --draws 8 --forecast-seed 2026102503 --power-solver continuous --q-min-bps 1000000 --reference-search candidates --out runs/q1_candidates_example
python run_allocation.py --n 10 --delta 0.1 --methods reference --blocks 10 --start-block 2 --draws 8 --forecast-seed 2026102503 --power-solver continuous --q-min-bps 1000000 --reference-search joint --out runs/q1_joint_example
```

这些reference行的预算字段是自身比较，不拿零预算违约当方法优越性证据。成功只说明预测约束与结构检查通过；真实速率与低速链路另报。
