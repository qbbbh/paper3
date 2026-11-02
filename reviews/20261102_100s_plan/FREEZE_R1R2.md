# R1/R2 冻结参数（2026-11-02）

**本文件在启动前写死。任何改动都必须新开一条记录，不得原地修改。**

## 1 主算法口径（v5-A）

| 项 | 值 | 依据 |
|---|---|---|
| 源码树 | `代码/开发_10车机制定型_v5_20261101/code` | `source_sha256 = 9723e1a7…` |
| 功率区间 | RF `[0.01, 20]` W、VLC `[0.1, 30]` W（**电输入**） | `p3base/config.py`，用户 2026-11-02 确认 |
| 静态功耗 | RF 0.1159 W、VLC 1.745 W | 同上 |
| EE 分母 | `P_c,mode + p`（电输入） | `link_electrical_power_w` |
| 主目标 | 逐链路 EE 的算术平均 | MODEL_SPEC 主目标 |
| 决策周期 | 100 ms，块内动作冻结 | MODEL_SPEC |
| 车辆/链路 | 10 车 / 5 条单向链路 | MODEL_SPEC |
| refine | `--maxmin-power-refine --maxmin-refine-solver nm` | v5-A（`v5_A_ref_nm_v5a` 同配置） |
| weak-link 修复 | **不启用**（`--maxmin-weak-link` 缺席） | 10 s 对照显示零增益且 2.4× 耗时 |
| 功率求解 | `--power-solver continuous` | 连续功率，无网格替代 |
| 参考搜索 | `--reference-search joint --joint-rounds 8` | 同 v5 全系列 |
| QoS | `--qos-metric chance --qos-chance-inflation 1.35 --qos-fallback-mean --qos-chance-rounds 2` | α=1.35 为校准值 |
| δ | `0.10` | MODEL_SPEC 待定项，本轮固定 |
| 轨迹数 | `--draws 32` | 同 v5 全系列 |
| 信息边界 | `--csi estimated --forecast-information trajectory` | 因果预测 |
| 块数 | `--blocks 999 --start-block 2` | ≈100 s |
| 线程 | `--threads 2` | 单 run |

## 2 对比算法（同批运行，保证完美配对）

在**同一条 run** 里同时产出三种方法，共享同一场景、同一预测、同一真值信道：

| 方法 | CLI | 身份 | 是否进主表 |
|---|---|---|---|
| `reference` | 默认 | Pure-EE 参考（效率优先上界） | ✅ Baseline 1 |
| `maxmin` | `--methods maxmin` | 本文方法 v5-A | ✅ Proposed |
| `literature` | `--methods literature`（默认 `dinkelbach`+`once`） | 文献复现（等效 SINR 关联 + 分式规划功率） | ✅ Baseline 2 |
| `random` | 不跑 | 随机下界 | ❌ 已有 100 s 数据，放辅助 |
| ideal / oracle | 不跑 | 信息条件诊断 | ❌ 已有 100 s 数据，放 6.4 |

**`literature` 的已知限制**：v4 `e10_literature_100s` 只有 **534/999** 块成功，
即文献基线在约 46% 的块上找不到满足 1 Mbit/s 的可行功率解。
若 v5 复现同样的低覆盖，**必须在正文如实报告"有效块/计划块"**，不得静默剔除。

## 3 完整命令

```powershell
$py = 'D:\anaconda\envs\pymarl\python.exe'
cd 代码\开发_10车机制定型_v5_20261101\code

# R1：设计种子
& $py -B -I run_allocation.py --n 10 --phase evaluate --split development `
  --blocks 999 --start-block 1 --draws 32 `
  --channel-seed 2026102401 --forecast-seed 2026102501 --learning-seed 2026102601 `
  --csi estimated --forecast-information trajectory `
  --power-solver continuous --reference-search joint --joint-rounds 8 `
  --qos-metric chance --qos-chance-inflation 1.35 --qos-fallback-mean --qos-chance-rounds 2 `
  --threads 2 --delta 0.10 `
  --methods reference maxmin literature `
  --maxmin-power-refine --maxmin-refine-solver nm `
  --out "<ABS>\paper3_100s\runs\R1_v5A_d010_100s"

# R2：验证种子（只改 channel-seed 与输出目录）
```

## 4 验收标准

| 项 | 要求 |
|---|---|
| 块数 | 每方法 999 个有效块，或**明确报告有效/计划** |
| 哈希一致 | 两 run 的 `source_sha256`、`config_sha256`（除 seed）、`mobility_sha256` 相同 |
| 三方法同批 | 同一 run 内三种方法逐块可配对 |
| 失败单列 | `fallback=True`、`coverage_failure=True` 的块必须单列，不计入通过率 |

## 5 不做的事

- 不改 v5 `code/`、不改 `source_sha256`；
- 不起 50/100 车；
- 不在本批启用 weak-link / SCA refine（留作 10 s 消融证据）；
- 不为凑数把 ideal/oracle/random 放进主对比表。

---

# 附：源码补丁记录（2026-11-02，**必须随论文档案保留**）

## 触发

R2 在 `block 588`（种子 `2026102801`）崩溃：

```
reference_power_search_failure: VLC 活动功率必须在 [0.1, 30] W
→ reference.success = False, reference.eta_ref = None
→ run_allocation.py:649  budget = (1-args.delta) * reference.eta_ref
→ TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'
```

**崩溃机制**：`run_allocation.py:634` 已经正确处理了"该方法自身失败"
（记 `block_failure` 并 `continue`）。但 `reference` 失败时，
`maxmin`/`literature` 这些**自身成功**的方法仍会走到 649 行，
用 `reference.eta_ref`（None）计算预算 ⇒ 整条 run 中断，且输出只在结尾写盘 ⇒ **已完成块全部丢失**。

## 补丁（唯一改动，位于 `p3learn`/`run_allocation.py` 方法循环内）

在 `action = decision.action` 之前插入与既有失败分支**语义完全相同**的守卫：

```python
if reference.eta_ref is None:
    rows.append(dict(common, row_type="block_failure", selected=False, active=False))
    block_records.append(dict(common, success=False, delivered_bits=0., data_energy_j=0.,
                              min_pair_ee_bit_per_j=0., mean_pair_ee_bit_per_j=0.,
                              mean_pair_ee_ratio_vs_actual_reference=0.,
                              network_ee_bit_per_j=0.))
    continue
```

**性质**：纯健壮性修复，不改变任何算法语义——
- 只在 `reference.eta_ref is None`（参考解失败）时执行；
- 执行内容与第 634–640 行处理 `not decision.success` 的方式一致；
- 参考解成功的块，代码路径逐字不变。

## 哈希变更（必须披露）

| 项 | 补丁前 | 补丁后 |
|---|---|---|
| `source_sha256` | `9723e1a7b1b0490e…` | **`3cab660907dff616…`** |
| 运行时 `config_sha256`（R1 配置） | — | `ecc42c784974f960…` |

**因此旧 R1（`9723e1a7…` 树上跑到 696/999）已作废并终止**，
四个 100 s run 全部在补丁树（`3cab6609…`）上重跑，保证主表内源码一致：

| ID | 目录 | 种子 | methods | refine |
|---|---|---|---|---|
| **R1'** | `runs/R1p_v5A_d010_100s` | 2026102401 | reference + maxmin + literature | **nm** |
| **R2'** | `runs/R2p_v5A_d010_100s_valseed` | 2026102801 | reference + maxmin + literature | **nm** |
| **R3'-1** | `runs/R3p1_norefine_d010_100s` | 2026102401 | reference + maxmin | **关闭** |
| **R3'-2** | `runs/R3p2_norefine_d010_100s_valseed` | 2026102801 | reference + maxmin | **关闭** |

## 已知副作用

- 种子 `2026102801` 的 block 588 会在主表中显示为**失败块**（无 `link` 行），
  按 `MODEL_SPEC.md`「求解失败块必须单列」处理，不计入任何通过率。
- R3' 与 R1'/R2' 的 `source_sha256` 相同，`config_sha256` 不同（refine 开关不同），这是预期的。
---

# 附二：块范围更正（2026-11-02，**第一次运行全部作废**）

## 错误

首轮用 `--start-block 2 --blocks 999`，覆盖 **block 2..1000**。
block 1000 的 `t = 100.0 s`，而块内细粒度真值评价还要往前推一个块（到 100.1 s），
场景记录只到 100.0 s，于是最后一个块抛：

```
ValueError: 时间 100.0005 不在记录范围 [0.0, 100.0]   (p3base/scene.py:59)
```

**后果**：4 个 run 都在跑到第 999 块时崩溃，且输出只在结尾写盘 ⇒ **全部作废**。

## 正确范围

已归档的 v4 100 s run（`e4_C`/`e5_D` 等）实测块范围是 **block 1..999**
（`min=1, max=999`），对应 `--start-block 1 --blocks 999`，即 t = 0.1..99.9 s。

| 项 | 错误值 | 正确值 |
|---|---|---|
| `--start-block` | 2 | **1** |
| 覆盖块 | 2..1000 | **1..999** |
| 块数 | 999 | 999（不变） |

## 影响

- 首轮四个 run（R1p/R2p/R3p1/R3p2）**全部作废**，输出目录已清空后重跑；
- **`source_sha256` 不变**（`3cab6609…`），只是参数更正；
- 与 v4 归档 100 s run 的块范围**现已一致**，可直接对齐比较。

## 教训

`--blocks 999` 与 `--start-block` 必须联合检查：**块数是跨度，不是上界**。
`scene.time_s` 长度是硬边界，超界只在最后一个块触发，因此**冒烟测试必须覆盖末尾块**，
不能只测前几块。