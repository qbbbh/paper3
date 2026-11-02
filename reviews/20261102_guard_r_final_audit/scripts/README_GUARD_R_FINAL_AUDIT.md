# Guard-R Final Audit v1（独立审计代码）

## 目的

本目录用于把 Guard-R 这条线做一次可复现、可审计的最终验证。它是**独立审计工具**，不是 v5 主算法的新版本。

本包不修改 `p3base/`、`p3learn/`，也不修改任何已归档 run。主脚本只从现有 v5 项目导入求解器、物理层、预测器和真值评价函数。

## 本版实际实现的比较

每个预登记 B/C 候选都执行以下四组求解：

| 组别 | 固定拓扑 | 求解器 | 用途 |
|---|---|---|---|
| `legacy_guard_r_candidate` | 候选拓扑 | 原 `maxmin_power_refine`（默认 NM） | 复现原 Guard-R |
| `legacy_same_topology_control` | 原执行拓扑 | 同一 NM、相同 maxiter/restarts | 旧方法的逐候选同拓扑控制 |
| `strict_feasible_first_candidate` | 候选拓扑 | QoS → 平均 EE 预算可行性 → Max-Min | 真正的候选配置内可行初始化 |
| `strict_same_topology_control` | 原执行拓扑 | 同一 Feasible-first 求解器、相同 evaluator-call cap | 严格等预算控制 |

两种 Feasible-first 组使用完全相同的 `max-evals-per-arm`、起点数量和三阶段预算分配。每次候选对应一次同拓扑控制，避免“一次控制对多个候选”的计算预算不一致。

## 三阶段求解逻辑

### Stage A：先找 QoS 可行点

通过 epigraph 形式最小化最大归一化 QoS 违反量：

\[
\min_{\mathbf p,t} t,\qquad
\frac{\widetilde R_k(\mathbf p)-q}{q}+t\geq 0,\quad t\geq 0.
\]

不把 QoS 违反量和 EE 预算违反量加权相加。

### Stage B：在 QoS 可行域内寻找预算可行点

\[
\max_{\mathbf p}\ \frac1K\sum_k \eta_k(\mathbf p)
\quad
\text{s.t.}\quad \widetilde R_k(\mathbf p)\geq q.
\]

随后检查是否达到：

\[
\frac1K\sum_k\eta_k(\mathbf p)\geq (1-\delta)\eta_{\rm ref}.
\]

这一阶段的局部搜索若没有找到预算可行点，只能报告“未找到”，不能报告“数学不可行”。

### Stage C：在两个约束同时满足的可行域内最大化最低链路 EE

\[
\max_{\mathbf p,z}\ z
\]

\[
\text{s.t.}\quad
\eta_k(\mathbf p)\geq z,\ \forall k,
\]

\[
\frac1K\sum_k\eta_k(\mathbf p)\geq(1-\delta)\eta_{\rm ref},
\quad
\widetilde R_k(\mathbf p)\geq q,
\quad
\underline p_k\leq p_k\leq\overline p_k.
\]

功率变量在对数域连续优化；没有离散功率网格。

## 评价口径

- 逐链路 EE：`mean_rate_bps[k] / link_electrical_power_w[k]`，分母使用项目的 `P_c + p`。
- 预算：均逐链路 EE，相对 `eta_ref`。
- QoS 预测约束：复用 `counterfactual_audit.QOS`，即项目的 chance 口径和 inflation。
- 真值评价：同一块真值信道、同一 100 ms 动作块。
- 同时检查两个真值预算：
  - `meanEE_truth >= (1-delta) * eta_ref_pred`
  - `meanEE_truth >= (1-delta) * eta_ref_true`
- 真值 QoS：所有通信对的块平均速率均不低于 `q_min_bps`。
- 合法真值改进：真值最低 EE 严格高于原执行动作，且真值 QoS、预测参考预算、真值参考预算全部通过。

## 放置位置（重要）

把以下文件直接放到现有 `counterfactual_audit.py`、`phase0_recoverability.py` 所在的目录：

`E:\博士\课题论文\第3篇-修改版\代码\分析_三条主线_20261101`

不要新建 `scripts` 子目录，也不要再额外套一层目录。原 `counterfactual_audit.py` 用 `HERE.parents[1]` 定位项目根目录，因此必须保持它原来的目录深度。

- `feasible_first_solver.py`
- `guard_r_final_audit.py`
- `merge_guard_r_final_audits.py`
- `test_feasible_first_solver.py`
- `test_guard_r_final_audit.py`
- `run_guard_r_seed.ps1`

预期结构：

```text
第3篇-修改版/
└─ 代码/
   ├─ 开发_10车机制定型_v5_20261101/
   │  └─ code/
   │     ├─ p3base/
   │     ├─ p3learn/
   │     └─ runs/
   └─ 分析_三条主线_20261101/
      ├─ counterfactual_audit.py       # 已有
      ├─ phase0_recoverability.py      # 已有
      ├─ feasible_first_solver.py      # 新增
      ├─ guard_r_final_audit.py        # 新增
      ├─ merge_guard_r_final_audits.py # 新增
      ├─ test_feasible_first_solver.py # 新增
      ├─ test_guard_r_final_audit.py   # 新增
      └─ run_guard_r_seed.ps1          # 新增
```

## 第一步：单元测试（不需要 SUMO 场景）

在 `分析_三条主线_20261101` 目录执行：

```powershell
D:\anaconda\envs\pymarl\python.exe -B test_feasible_first_solver.py
```

运行两组测试：

```powershell
D:\anaconda\envs\pymarl\python.exe -B test_guard_r_final_audit.py
```

共 8 项合成测试，覆盖：
1. 能找到 QoS 和预算可行点，并继续执行 Max-Min；
2. QoS 可行但预算点未找到时，不错误声称数学不可行；
3. QoS 可行点未找到时，单独报告；
4. 拒绝非法功率边界；
5. Phase-0 B/C 预登记筛选规则；
6. 接收端/模式或连续功率变化判定；
7. 跨拓扑参考功率映射保持模式功率边界。

也可以检查命令行参数：

```powershell
D:\anaconda\envs\pymarl\python.exe -B -I guard_r_final_audit.py --help
D:\anaconda\envs\pymarl\python.exe -B -I merge_guard_r_final_audits.py --help
```

## 第二步：生成同 seed 的 Phase-0 预登记文件

`phase0_recoverability.py` 当前没有把 channel seed 写进 JSON 元数据。因此，后续审计命令要求显式提供 `--phase0-seed`，由操作者确认它与生成 Phase-0 的 `--channel-seed` 相同。

设计种子示例：

```powershell
D:\anaconda\envs\pymarl\python.exe -B -I phase0_recoverability.py `
  --blocks 100 `
  --run "E:\博士\课题论文\第3篇-修改版\代码\开发_10车机制定型_v5_20261101\code\runs\v5_C_ref_sca_s_v5a" `
  --channel-seed 2026102401 `
  --out "out_phase0_design"
```

验证种子必须使用**项目中已经确定的独立验证 seed**，不要临时编造或挑选 seed：

```powershell
D:\anaconda\envs\pymarl\python.exe -B -I phase0_recoverability.py `
  --blocks 100 `
  --run "<对应验证 run 目录>" `
  --channel-seed <已冻结的验证 seed> `
  --out "out_phase0_validation"
```

## 第三步：先做 smoke test

先用一个选中块、一个候选和较小评价预算检查完整接口：

```powershell
D:\anaconda\envs\pymarl\python.exe -B -I guard_r_final_audit.py `
  --run "<v5 run 目录>" `
  --phase0 "out_phase0_design\phase0.json" `
  --channel-seed 2026102401 `
  --phase0-seed 2026102401 `
  --role smoke `
  --out "out_guard_r_smoke" `
  --max-blocks 1 `
  --max-candidates-per-block 1 `
  --max-evals-per-arm 120 `
  --starts 3 `
  --maxiter-qos 25 `
  --maxiter-mean 30 `
  --maxiter-maxmin 35 `
  --legacy-maxiter 40 `
  --legacy-restarts 1
```

注意：`--max-blocks 1` 是“第一个命中预登记规则的块”，不是固定 Block 1。脚本仍会按顺序推进 Session，以保持信道估计器的时间状态。

Smoke 应检查：
- reference / maxmin / 候选动作是否与归档一致；
- strict solver 能否调用预测评价器；
- 输出中是否分别记录两种真值预算；
- 预测器调用数是否不超过每 arm 上限；
- v5 原始源码哈希是否记录。

Smoke 通过后再启动完整实验。

## 第四步：设计种子与验证种子

设计种子完整审计：

```powershell
D:\anaconda\envs\pymarl\python.exe -B -I guard_r_final_audit.py `
  --run "<设计种子对应 v5 run>" `
  --phase0 "out_phase0_design\phase0.json" `
  --channel-seed <设计 seed> `
  --phase0-seed <同一个设计 seed> `
  --role design `
  --out "out_guard_r_design" `
  --max-evals-per-arm 600 `
  --starts 5
```

独立验证种子完整审计：

```powershell
D:\anaconda\envs\pymarl\python.exe -B -I guard_r_final_audit.py `
  --run "<验证种子对应 v5 run>" `
  --phase0 "out_phase0_validation\phase0.json" `
  --channel-seed <验证 seed> `
  --phase0-seed <同一个验证 seed> `
  --role validation `
  --out "out_guard_r_validation" `
  --max-evals-per-arm 600 `
  --starts 5
```

完整运行可能较慢。先用 smoke 测量单候选耗时，再决定是否启动全量。

## 第五步：合并设计与验证结果

```powershell
D:\anaconda\envs\pymarl\python.exe -B -I merge_guard_r_final_audits.py `
  --design "out_guard_r_design\audit.json" `
  --validation "out_guard_r_validation\audit.json" `
  --out "out_guard_r_combined"
```

合并脚本会检查：
- 两个 seed 不同；
- 一个文件标记 `design`，另一个标记 `validation`；
- 两轮 v5 源码树 hash 一致；
- delta 和 QoS 阈值一致。

## 输出文件

每个 seed 的输出目录包括：

- `audit.json`：完整元数据、逐候选、逐 arm、逐求解阶段诊断。
- `candidates.csv`：便于 Excel 汇总的候选级结果。
- `report.txt`：简要统计与结论边界。

合并目录包括：

- `combined_summary.json`
- `paired_candidates.csv`

## 判读规则

本工具不会把“局部优化没找到”写成“配置不可行”。

只允许以下结论：

- `qos_feasible_point_not_found`：当前多起点搜索没有找到 QoS 可行点；
- `qos_feasible_budget_point_not_found`：找到 QoS 可行点，但当前预算内没有找到满足平均 EE 预算的点；
- `budget_feasible_point_found_maxmin_not_improved_or_not_completed`：找到了同时满足约束的点，但 Max-Min 阶段没有产生被接受的局部改进或未完成；
- `maxmin_local_solution_found`：找到了满足约束的局部 Max-Min 解。

“未找到”均不是全局不可行证明。

预算上界 `budget_ceiling_screen.py` 本轮不作为硬剪枝器使用，因为其 40 点功率采样最大值不是连续域的认证上界。它最多只能作候选排序诊断。

## 冻结约束

- 不修改 v5 主算法；
- 不改变 100 ms 块级动作；
- 不改变 RF/VLC 物理模型、功率边界、静态电耗；
- 不改变逐链路 EE 分母 `P_c + p`；
- 不改变项目既定 delta、QoS 门限和 chance inflation；
- 不用离散功率网格代替连续功率优化；
- 不把预测侧通过等同于真值侧通过；
- 不删除失败块、求解失败或重建不一致记录。
