# 实验归档：Tag `20261031g`（v4 口径主算法最终批）

**归档日期**：2026-10-31
**内容**：10 车、100 块、v4 功率口径下主算法与全部对比算法的 23 个 run（691 MB）、
产生它们的代码快照、启动脚本、日志、分析脚本与记录文档。**这是论文 §13–§16 引用的那批数据。**

## 1. 目录结构

```
实验归档_20261031g_主算法v4/
├─ README.md                     本文件
├─ MANIFEST_sha256.txt           全部 219+ 个文件的 sha256（含 run 内 5 个文件）
├─ 口径v4与主算法批量运行记录_20261031.md   完整记录（名词表、每组做什么、实测、边界、更正）
├─ code/                         代码快照（与产生 run 的源码逐字节相同）
│   ├─ p3learn/  p3base/  tests/ 顶层 .py
│   ├─ _备份_v3_20261031/         v3 备份（source_sha256 把它也算在内，必须保留）
│   └─ runs/                      **23 个 run**（从主树移入，不再有副本）
├─ launch/                       5 个批次启动脚本（run_all / e5_refine / e4 / delta_scan / final_batch …）
├─ logs/                         本批控制台与分阶段日志
└─ analysis/                     分析/审计/对比脚本 + 本批证据
    ├─ audit_batch.py  audit_after_move.txt   移动后重新审计的结果（23/23 通过）
    ├─ cmp_bg.txt                               与 Tag 20261031b 的逐位复现比对（0.000000%）
    └─ compare_final.py  compare_tags.py  portfolio_*.py  best_variant.py …
```

## 2. 23 个 run 分别是什么

| 类别 | run 目录 | 方法 | 用途 |
|---|---|---|---|
| 主实验（e5，有 refine） | `e5_B_mean_d005` / `e5_B_mean_d010` | reference + maxmin | mean QoS，δ=0.05/0.10 |
| 主实验（e5，有 refine） | `e5_D_chance_d005` / `e5_D_chance_d010` | reference + maxmin | **chance QoS，δ=0.05/0.10（主配置是 d010）** |
| 消融（e4，无 refine） | `e4_A_mean_ref`、`e4_C_chance_ref` | reference | 两条基线 |
| 消融（e4，无 refine） | `e4_B_mean_maxmin_d005/d010`、`e4_D_chance_maxmin_d005/d010` | reference + maxmin | 有无公平性模块的对照 |
| 独立信道 | `e6_B_ref`、`e6_B_d005`、`e6_B_d010` | reference + maxmin | channel seed `2026102801` |
| δ 曲线 | `e6_A_chance_d020`、`e6_A_chance_d100` | reference + maxmin | δ=0.20 与纯 max-min |
| 随机下界 | `e7_random_chance_d010` | random | 均 EE 只有 3.4%，QoS 0% |
| genie CSI | `e7_ideal_ref`、`e7_ideal_chance_d010` | reference + maxmin | `--csi ideal`，**不是上界**（实测更低） |
| 非因果 oracle | `e9_oracle_bound` | reference + maxmin | 动作池上界也用它 |
| 文献复现 | `e10_literature`、`e10_lit_nom001`、`e10_lit_nom300`、`e10_lit_cont` | literature | 4 个变体；前三个各 16 块预期失败 |

每个 run 目录内 5 个文件：`summary.json`（汇总）、`results.csv`（逐链路）、
`fine_samples.npz`（抽样）、`continuous_power_audit.json`（连续功率求解审计，约 26 MB）、
`training_curve.json`。

## 3. 怎么验（在这台机器上）

```powershell
# 1) 单个 run 的完整审计（源码哈希必须等于本目录 code/）
cd 实验归档_20261031g_主算法v4\code
D:\anaconda\envs\pymarl\python.exe -B -I audit_allocation_outputs.py --require-current-source runs\e5_D_chance_d010_20261031g

# 2) 23 个 run 一次过
D:\anaconda\envs\pymarl\python.exe -B -I analysis\audit_batch.py code analysis\audit_after_move.txt "runs/*_20261031g"

# 3) 逐文件哈希
sha256sum -c MANIFEST_sha256.txt
```

## 4. 口径（跟着数据走，不要在别的口径下解读这些数字）

- `EE_k = r_k / (P_c + p_k)`，再对 k 取算术平均；`p` 是**电输入功率**。
- 静态功耗：RF 0.1159 W / VLC 1.745 W；动作区间 RF 0.01–20 W、VLC 0.1–30 W。
- `REFERENCE_VERSION = "pure-mean-electrical-ee-joint-mode-power-v4"`（主路径带 `+continuous-qt-sca-v1`）。
- 参考网格 `(0.01, 0.3, 3.0)`；`source_sha256 = d13dd9e0…`。
- **v3 口径下的绝对 EE 不能与这里并列**，跨口径只比相对提升。

## 5. 与主树的关系

主树（`代码/开发_10车机制定型_v4_20261027/`）仍然保留：`code/` 源码、`run_*.ps1` 启动脚本、
`logs/`（含本批日志副本）、`_staging_20261031/` 分析脚本、其余 83 个历史 run（Tag 20261027–f）。

**没有搬过来的**：`20261027/20261028/20261031/b/c/d/e/f` 这 83 个 run（约 2.3 GB），
以及 `snapshots/`（656 MB，20261028 的诊断快照）。

## 6. 已知限制

1. `analysis/compare_tags.py 20261031b 20261031g` **在本目录里跑不出 18 组**：b 批次还在主树。
   若要复现 §16.2，需要把主树的 `*_20261031b` 复制到本目录 `code/runs/` 再跑，或直接看已归档的
   `analysis/cmp_bg.txt`（结论：18 组、30 个 (组,方法) 对、`max|Δ| = 0.000000%`）。
2. `source_sha256` 覆盖 `code/**/*.py` 的**全部**文件，包括 `_备份_v3_20261031/`。
   删掉任何 `.py`（含备份）都会让审计报 `source_sha256 does not match`。
3. `summary.json` 里记录了运行时的绝对路径（`E:\…\第3篇-修改版`），那是历史记录，
   不代表文件现在还在那个位置。
