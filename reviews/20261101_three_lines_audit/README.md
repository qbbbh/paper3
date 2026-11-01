# 2026-11-01 三条主线的数值与结论（供独立复审）

**复审任务**：核对下面 §3 的每条结论是否真的由 §2 的数据支持。
请特别复核 §4（已作废/更正条目）与 §6（未建立的东西）——如果这两节与 §3 冲突，以它们为准。

---

## 1 这个包覆盖什么

把 Paper3 的三条主线
> 估计极限（CRLB）→ 可靠服务资源代价 → 代价引导联合更新

从"研究构思"推进到**可被数据检验的命题**，并给出实测：

| 主线 | 本轮做到什么 | 结论强度 |
|---|---|---|
| 一 估计极限 | 分位数校准桥（命题 1）+ α 反解（推论 1.1），实测有数 | ✅ 可写进论文 |
| 二 服务代价与切换 | 结构命题（引理 1、命题 2′、3、4）+ 反事实真值审计 | ✅ 结构成立，原命题 2 被否证 |
| 三 候选评价与联合更新 | 排序有效性（命题 6）+ 守卫价值的反事实量化 | ✅ 可写，且是负结果为主 |

---

## 2 数据文件与可复现层级

### 2.1 文件清单

| 文件 | 内容 | 大小 |
|---|---|---|
| `data/calib_100s_999blocks.json` | `e5_D_chance_d010_20261031h`（999 块 / 9990 条链路）的覆盖率与 α 反解 | 2.4 KB |
| `data/calib_gate_weak_10s_v5C.json` | `v5_C_ref_sca_s_v5a`（100 块 / 1000 条链路）的覆盖率、α 反解、门排查、候选族统计 | 4.7 KB |
| `data/routeB_summary.json` | 反事实审计的逐块逐候选记录（含真值） | 276 KB |
| `data/routeB_blocks.csv` | 每块一行的汇总（弱链路三个功率、命题 2 前提、τ、强制采纳） | 22 KB |
| `data/routeB_run_log.txt` | 审计的逐块控制台输出 + 汇总报告 | 12 KB |
| `PROPOSITIONS_three_lines.md` | 命题与证明正文（引理 1、命题 1–7、边界） | 34 KB |
| `verify_claims.py` | **把本文 §3 的每条结论变成可执行断言**，只读 `data/`，不需要 v5 树 | 7.4 KB |
| `VERIFY_RESULT.txt` | `verify_claims.py` 的实际输出：**34 条检查，PASS 34，FAIL 0** | 2.1 KB |
| `scripts/diagnose_three_lines.py` | 生成 `calib_*.json`（纯 CSV/JSON，不重跑仿真） | 19 KB |
| `scripts/counterfactual_audit.py` | 生成 `routeB_*`（重建决策 + 真值重打分） | 27 KB |

### 2.2 两级可复现性（请据此判断证据强度）

**A 级（任何人都能算）**：`calib_*.json` 里的每个数都能由源 `results.csv` 的列
`predicted_rate_bps / predicted_p05_bps / actual_rate_bps / csi_age_s` 重新算出。
算法写在 `scripts/diagnose_three_lines.py` 的 `coverage_table` 与 `solve_alpha`，正文 §3 给了公式。

**B 级（需要审计方的代码树，本仓库没有）**：`routeB_*` 需要
`开发_10车机制定型_v5_20261101/code/`（含 `p3base`/`p3learn`/`run_foundation.py`）与
10 车 SUMO 场景。**本仓库没有这棵 v5 树**，所以复审方无法直接重跑 `counterfactual_audit.py`。
为此该脚本自带两道自检（§5），并**逐块保存了预测值与真值**，使结论可以在 `routeB_summary.json`
里逐条验算而不必重跑。

### 2.3 本仓库源码基线 ≠ 被审计的源码树

仓库基线是 `95960f5a…`（`code_10veh_v4/`）。本轮审计的对象是
`开发_10车机制定型_v5_20261101/code/`，`source_sha256` = `d13dd9e0…` 之后的 v5 树，
**不在本仓库内**。两者不可混用。

---

## 3 结论清单（逐条绑定数据）

编号规则：`C` = 结论（claim），`R` = 已作废/更正（retracted）。

### 主线一

**C1（成立）** 预测器的名义 5% 分位被实际跌破的频率是 **10.27%**（名义 5% 的 **2.05 倍**），n = 9990。
- 数据：`data/calib_100s_999blocks.json` → `parts.calib.overall.violation_rate = 0.1027`、`n = 9990`
- 定义：`violation = 1[actual_rate_bps < predicted_p05_bps]`，只在 `predicted_p05 > 0` 且 `predicted_rate > predicted_p05` 的行上统计
- 复核要点：这是"分位数偏窄"，即预测方差被低估；**不是**"估计器不准"

**C2（成立）** α 可由覆盖率反解，`α* = 1.3003`；实现值 1.35 时跌破率 **4.58%**，即实际可靠性 **95.42%**。
- 数据：同文件 → `parts.alpha.alpha_star = 1.30029296875`、`trace` 中 `α=1.35 → 0.0458`
- 算法：`R̃(α) = mean·(p05/mean)^α`，二分求最小 α 使跌破率 ≤ 0.05（`solve_alpha`）
- 第二数据源：`data/calib_gate_weak_10s_v5C.json` → `parts.alpha.alpha_star = 1.2314453125`（10 s 窗口）

**C3（成立）** 因此 α = 1.35 是**校准值**而非任意超参：两个独立窗口给出 `α*` ∈ [1.23, 1.30]，
实现值 1.35 偏保守 4–10%。
- 复核要点：若只采信一个窗口，`α*` 会偏向 1.23 或 1.30；**不要引用单点值作为定论**

**C4（成立，且是一条局限）** 按 `csi_age_s` 分层**不能**分离"年龄"与"模式"：
- `data/calib_100s_999blocks.json` → `parts.calib.by_csi_age` 显示 `age<5ms` 违约率 10.73% > `5–50ms` 8.96% > `50–100ms` 9.10%，**年龄越大反而越好**；
- 但 `parts.calib.by_mode` 显示 RF 10.73%（n=7347）、VLC 9.00%（n=2643），与年龄分层是同一批样本。
- 原因：RF 导频周期 1 ms、VLC 50 ms，`csi_age` 被模式决定。**不能据此写年龄效应**。

### 主线二

**C5（成立，且否证原命题 2）** "瓶颈链路被 QoS 钉住"的前提 `p_ee < p_svc` 在 100 块中
**一次都没成立（0/100）**。
- 数据：`data/routeB_blocks.csv` 列 `premise_holds`（全为 `False`）、`p_cur_w`、`p_svc_w`、`p_ee_w`
- 定义：`p_svc` = S1 服务功率（`service_power_probe`），`p_ee` = 该链路自身能效峰值功率（`link_ee_optimum`），`p_cur` = 执行功率
- 复核要点：`p_cur > p_svc` 占 **95/100**（列 `over_powered`），与外部复核一致；但"70/100 块 `p_ee < p_svc`"**无法复现**

**C6（成立，替代命题）** 更准确的结构是**峰值钉住**：`p_cur / p_ee` 中位 **0.98**
（60 块在峰下、40 块在峰上）。
- 数据：`data/routeB_blocks.csv` 的 `p_cur_w / p_ee_w` 逐行相除
- 判定：`p_cur / p_svc` 中位 8.26、`p_ee / p_svc` 中位 10.96
- 由此：沿最弱链路自身功率**任意方向**移动都会降低其 EE（引理 1 单峰性）

**C7（成立，最强的一条证据）** A 类候选（把最弱链路功率抬到 `p_ee`）在真值上是**空操作**：
Δmin 中位 **−0.00%**，且 **15/100 块逐位相同**（|Δ| < 1e-6）。
- 数据：`data/routeB_summary.json` → `summary.by_stage.A.d_min_median ≈ -0.0`；
  逐块验证用 `blocks[i].candidates[*].d_min_vs_exec_pct`
- 意义：这与 C6 独立互证——已经在峰值上，再抬不会变

**C8（成立）** B 类候选（换接收端）被淘汰得最彻底：Δmin 中位 **−97.71%**，只有 6/102 更好。
- 数据：`summary.by_stage.B = {n:102, better:6, d_min_median:-97.71, d_mean_median:-35.27}`

**C9（成立）** C 类（换模式）与 S2 类（降功率救人）多数**更差**：Δmin 中位分别 −27.62% / −45.34%。
- 数据：`summary.by_stage.C`、`summary.by_stage.S2`

### 主线三

**C10（成立）** 候选内的预测排序与真值排序**正相关但有限**：Kendall τ 均值 **+0.3576**，
正相关块 **45/66（68.2%）**。
- 数据：`summary.kendall_tau_mean = 0.35757575757575755`、`summary.kendall_tau_positive_frac`
- 注意 n = 66：只有候选数 ≥ 3 的块才有 τ

**C11（成立）** 守卫（预算门 + QoS 门）**基本无损**：每族真值 Δmin 中位 ≤ 0（C5–C9），
且 100 块中只有 **28 块**存在"真值更优**且**真值仍在 δ 预算内"的被拒候选。
- 数据：由 `routeB_summary.json` 逐块筛 `candidates[*].truth_min_ee > executed.min_ee` 且
  `candidates[*].truth_budget_ok == true`
- 该类块强制采纳最好候选的 min EE 增益：**中位 +0.02%**、均值 +12.85%、
  仅 **1 块 > 10%**、**1 块 > 50%**（28 块里 24 块是 A，属噪声级）

**C12（成立，回应"全都采纳"这一思路）** 把每块**预测** min EE 最高的候选强制采纳，
比守卫更差：**35/100 块真值更优，但均 EE 中位 −1.96%，且仅 47% 的块真值仍在预算内**。
- 数据：`summary.force_accept = {n:100, wins_on_truth_min:35, d_min_median_pct:-0.0, d_mean_median_pct:-1.96, truth_budget_ok_frac:0.47}`

---

## 4 已作废 / 已更正的条目（**复审请重点看这一节**）

这些是我在本轮中先说、后被自己的数据推翻的说法。保留是为了让复审方能判断我有没有隐藏错误。

**R1（作废，最重要）** 我曾报"79% 的块存在真值更优的被拒候选，中位增益 **+87.9%**"。
**该数字作废**：当时用 `block.summarize_block` 的 `mean_pair_ee_bit_per_j`，它是
`rate / p` 的旧口径，而论文口径是 `rate / (P_c + p)`。同一块因此算出 3032 M（旧）vs 371.8 M（正确）。
修正后的量级见 **C11**：**28% 的块、+0.02% 中位增益**。

**R2（否证）** "基准解本身违反 QoS，于是候选继承违约、门在构造上不可满足"。
→ `data/calib_gate_weak_10s_v5C.json` → `parts.gate`：`reference` 与 `maxmin` 各
**100/100 块全达标**、链路违约率 0。

**R3（否证）** "α = 1.35 过大，把 QoS 门压死"。
→ 见 C2/C3：`α* = 1.23–1.30`，1.35 只偏保守 4–10%，对应可靠性 95.4%。

**R4（否证）** "命题 2 的瓶颈钉住（`p_ee < p_svc`）能解释 0 采纳"。
→ 见 C5：前提 0/100，重写为峰值钉住（C6）。

**R5（无法复现的外部说法）** "95/100 块当前功率 > 服务功率，其中 **70 块**自身 EE 峰值 < 服务功率"。
前半句与 C5 一致（95/100），**后半句 0/100**。两边用的是同名函数
（`service_power_probe` / `link_ee_optimum`），差异只能来自"最弱链路"的选取口径。
论文引用前必须统一。

**R6（口径误读）** 我曾把 `weak_link.json` 的 `qos_violation_links` 当作"QoS 门"统计，
得出"100% 候选被 QoS 门拒"。该字段在产生这批 run 的版本里是**诊断字段、不参与拒绝**
（同一记录里 `"fallback": false` 即可否证）。`data/calib_gate_weak_10s_v5C.json` 的
`parts.weak.by_family.*.QoS通过` **沿用了我当时的错误定义，请勿引用该列**；
正确做法是看 `accepted` 字段（全为 false）与 `protect_budget` 的返回。

---

## 5 两道自检（判断 `routeB_*` 可信度的依据）

**S1 决策复现校验**：审计不信任自己的复现。每块把重建出的最终动作与
`results.csv`（`method=maxmin`）逐项比对链路身份、模式、功率，并把每个候选的功率向量
与 `weak_link.json` 的 `powers` 逐元素比对（atol 1e-6）。

- 结果：`100/100` 块 `mismatch = None`，`blocks_excluded = 0`（`summary.blocks_verified = 100`）

> **已知的复现前提**：归档 v5 run 的 `summary.json` 记 `vlc.sinr_model = "square_law"`，
> 但当前 `p3base/config.py` 默认是 `"linear"`，且 `run_compare_10s_v5.ps1` **没有** `--vlc-sinr`。
> 审计里显式覆盖成 `square_law` 才能复现。用默认值会得到 block 2 的
> `eta_ref = 680.404 M`，而归档是 **445.884 M**（差 53%）。这本身是 C13。

**C13（成立，是被审计代码的缺陷）** **v5 归档无法用当前启动脚本复现。**
- 证据：block 2 的 `reference.eta_ref`：归档 445.884 M；默认 `linear` 680.404 M；
  显式 `square_law` 445.884 M（且 `min_pair_ee = 34.168 M` 与链路模式
  `[VLC, RF, RF, RF, RF]` 全部逐位吻合）
- 根因：`run_compare_10s_v5.ps1` 缺 `--vlc-sinr square_law`；而 `source_sha256` 只覆盖
  `code/**/*.py`，改 `.ps1` 不改变哈希，所以"审计通过"与"复现不出"可以同时为真

**S2 真值口径自检**：每块把审计算出的真值 EE 与归档 `results.csv` 的块级
`min_pair_ee_bit_per_j` / `mean_pair_ee_bit_per_j` 比对，偏差 > 1e-6 即把该块标为排除。

- 结果：`summary.truth_ee_max_rel_dev = 0.0`（100 块全部逐位一致）
- 这条自检正是发现 R1 的原因

---

## 6 没有建立的东西（**不要当成已证明**）

1. **引理 1（单链路 EE 单峰性）** 有完整解析证明（`PROPOSITIONS_three_lines.md` §2.1，两种 SINR 口径都证了），
   但这是把已知形状写清楚，**不是新定理**。
2. **命题 3a（跨模式切换闭式判据 `C_{ℓ'} < C_ℓ`）** 只有证明，**没有独立实验检验**。
   本轮 C 类候选混了同模式换接收端与跨模式换功率，不能当它的验证。
3. **路线 C（受控合成相图）未做**——所以"补功率 / 重构"的边界**没有画出来**，
   也没有正例/反例配对。
4. **单一信道实现**（`channel_seed = 2026102401`）、**单一 SUMO 轨迹**、**10 车**。
   所有结论不外推到 50/100 车或其它衰落实现。
5. **τ = +0.358 是"部分相关"**，不能据此说代理评分"有效"，也不能说它"无效"。
6. **CRLB 本身不是本文创新**：命题 1 是分位数对尺度参数的单调性（标准工具），
   本文的贡献在"桥 + 校准"。
7. **不能写"全局最优""真实 QoS 保证""100 ms 在线"**——理由见
   `PROPOSITIONS_three_lines.md` §7。

---

## 7 复现命令

### 7.1 零依赖：校验本文的每一条结论（**请先跑这个**）

不需要 v5 源码树、不需要场景数据、不重跑仿真，只读 `data/`：

```powershell
cd reviews\20261101_three_lines_audit
D:\anaconda\envs\pymarl\python.exe -B -I verify_claims.py
# 期望：共 34 条检查，PASS 34，FAIL 0（实际输出见 VERIFY_RESULT.txt）
```

该脚本对每条结论断言"期望值 vs 实测值"，任何一条 FAIL 都说明本文与数据不符。
其中包含两项交叉一致性检查：`blocks.csv` 与 `summary.json` 的前提数一致、
真值 EE 与归档块级 EE 的最大相对偏差 < 1e-9。

### 7.2 重跑（需要 v5 源码树，本仓库没有）

```powershell
cd 代码\分析_三条主线_20261101
$py = 'D:\anaconda\envs\pymarl\python.exe'

# A 级：覆盖率 + α 反解 + 门排查 + 候选族统计（纯 CSV）
& $py -B -I diagnose_three_lines.py --part all `
    --run "..\开发_10车机制定型_v5_20261101\code\runs\v5_C_ref_sca_s_v5a" `
    --out out_review

# B 级：反事实审计（需要 v5 源码树 + 10 车场景；约 38 min）
& $py -B -I counterfactual_audit.py --blocks 100
```

`counterfactual_audit.py` 会把结果写到 `out_routeB/`，并打印 §3 的全部汇总。
`diagnose_three_lines.py` 不会改动任何 `code/` 下的文件，**不改变 `source_sha256`**。

---

## 8 一句话给复审方

> 本轮真正的正向结果只有两类：**分位数校准（C1–C3）** 与 **守卫无损（C11）+ 排序有效但有限（C10）**。
> 其余多是**否证与更正**（R1–R6、C5、C13）。如果你发现 §3 里任何一条与 §4/§6 冲突，请以 §4/§6 为准并直接指出。
