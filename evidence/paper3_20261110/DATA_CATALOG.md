# Paper3 20261110 数值证据包

这份资料用于核查2026年11月10日新增连续功率协调阶段后的结果。三组各有1000个计划周期，周期长度0.1秒。包含全部周期、原失败、全部现存正式对比方法和同拓扑消融，不按结果好坏筛选。

## 建议阅读顺序

1. [主方法完整报告](reports/10车与50车_功率改进全量核查_20261110.md)和[同拓扑完整报告](reports/同拓扑消融_补算复用与全量结论.md)。
2. [独立审计](audits/改进求解器独立审计_20261110.md)与[审查问题](REVIEW_QUESTIONS.md)。
3. [字段说明](tables/SCHEMA.json)、[配置与种子](tables/GROUP_CONFIGS.json)、[导出计数](tables/EXPORT_COUNTS.json)。
4. 逐周期表、逐链路表；需要从信号和干扰重算公式时使用固定验证数组。

## 文件与范围

| 目录/文件 | 内容与用途 |
|---|---|
| `reports/` | 主方法和同拓扑完整报告、原聚合JSON、同拓扑逐周期表、全部3000周期复用溯源 |
| `audits/` | 原独立审计报告、数组机器审计、原主方法/同拓扑输入等同性核查；历史脚本仅保留来源、哈希及固定提交地址 |
| `tables/{10A,10B,50B}_periods.csv` | 每组8000行：8个方法各1000个计划周期，包含失败；无动作时EE留空 |
| `tables/{10A,10B,50B}_links.csv.gz` | 全部成功动作的逐链路配置、输入功率、辐射/光功率、预测均值与下分位、实际周期均值速率、EE和检查结果 |
| `tables/*_links_preview.csv` | 固定周期100的可直接阅读预览；不是全量结果的替代品 |
| `tables/*_replay_records.jsonl.gz` | 每组2000条逻辑输出记录，含求解诊断、预测精确复现检查、真值评价和恢复/新增耗时；同拓扑复用显式注明原始来源 |
| `samples/` | 固定100、200、…、900周期，三组、主方法和同拓扑两个逻辑方法的原始NPZ/record/commit；54份NPZ合计不足10MB |
| `manifests/` | 本次主方法6个运行分片及同拓扑3个运行的原始manifest，保存配置、代码和输入哈希 |
| `baseline_records/` | 原10A、10B、50B主批、50B补跑的完整原始results.csv和summary.json，无损gzip；含所有现存方法 |
| `baseline_records/pre_guard_cross_scale/` | **新增功率协调阶段之前**的跨规模报告、诊断与统计，数值未改；说明改进动机，不能作为新版主方法结果 |
| `SOURCE_MANIFEST.json` | 本地来源、原文件哈希、发布路径及发布文件哈希；本地绝对路径用于追溯，不表示GitHub可直接访问 |
| `verify_package.py` | 仅依赖本包与NumPy的只读核查：哈希、全部周期与逐链路公式、固定数组的SINR/速率/EE、原始gzip内容 |

当前完整正式数据内旧方法为reference、maxmin、same_topo_maxmin、no_refine、Aboagye、Zappone六类；没有单独random方法记录。不得把旧历史开发实验混入本轮作为“随机对比”。

10A与10B使用同一10车交通数据、不同信道种子；50B是50车场景。A/B不是RF与VLC两种模式。具体种子和数据哈希在配置、manifest中。50B主方法和同拓扑均已完成全部1000周期的结果覆盖。

## 定义、分母与单位

- 优化功率是发送支路电输入功率。RF辐射功率为0.1倍输入功率，VLC光功率为0.25倍输入功率。参数范围和电路功率以各组配置为准。
- 单链路EE为实际或预测周期平均速率除以“电输入功率+模式电路功率”。周期平均EE是活动链路EE的算术平均；周期最低EE是该周期最弱链路的EE。不是总速率/总功率。
- EE报告采用成功周期；方法间成对比较取共同成功周期。执行服务采用“成功得到动作且所有链路实际达标”的计划周期比例，分母1000。失败EE为空而非零。
- 有些独立对比方法会在初始分配失败的周期仍得到动作；该周期相对初始分配的损失、预算检查为空，不删除其成功结果。
- `predicted_mean_budget_check_pass`是预测均值相对预测初始均值的数值检查；只有`budget_constraint_applies=True`的方法把它作为优化约束。真实损失相对同周期初始动作的实际平均EE另算。
- 初始均值后备、最终返回初始动作、最终下分位检查、无动作失败分别保存。`retained_previous_incumbent`表示保留旧主方法/旧同拓扑结果，不等于返回初始动作。
- 速率bit/s、EE bit/J、功率W、耗时s。转换为Mbit/J需除以10^6。CSV空值表示缺失或不适用。
- 完整新版时间为历史完整决策耗时加新增功率阶段耗时的组合估算，并非从头运行新版全流程的实测。预测重建和真值评价不混入新增优化耗时。

## 数组可以核查什么

每个样例含reference/old/new三套预测速率样本及细时间步signal/interference/noise/SINR/rate。`tx/rx/mode/old_power_w/new_power_w`属于old/new动作；reference动作的拓扑和功率要从同组逐链路表中选`method=reference`读取，不能套用old/new拓扑。

这些样例足以核查保存动作的速率公式、周期聚合、EE、分位校准和约束检查。它们**没有完整条件信道轨迹库**，不能凭当前动作的速率样本任意重评另一个功率。完整原始交通、估计恢复点和全量细粒度轨迹仍保存在本地；此包为紧凑审核材料，未把“未上传”写成“不存在”。完整重启数据没有随包复制。

样例集合是改进开发前预定的验证周期，不是从最终结果挑选。全量逐周期和逐链路数据同时保留，因此不限于样例判断数值效果。

## 读取与验证

```python
import csv, gzip, json
from pathlib import Path
root = Path('evidence/paper3_20261110')
with gzip.open(root / 'tables/50B_links.csv.gz', 'rt', encoding='utf-8', newline='') as f:
    rows = list(csv.DictReader(f))
with gzip.open(root / 'baseline_records/50B_main/summary.json.gz', 'rt', encoding='utf-8') as f:
    original_summary = json.load(f)
```

在仓库根目录运行`python evidence/paper3_20261110/verify_package.py`即可只读核查，不会训练或仿真。这是证据包当前唯一的Python入口；它仅依赖标准库、NumPy和包内数据。旧审计、旧报告生成器与一次性导出工具已从当前目录移除，原脚本哈希及移除记录仍在SOURCE_MANIFEST中，并可从[固定历史提交](https://github.com/qbbbh/paper3/tree/6ae23c04c091edee70e228cae39d72efb2af04a8/evidence/paper3_20261110)追溯。此项清理不改变原审计结论和任何数值证据，也不表示仅凭本包即可重做全部优化。

Markdown报告只改写发布链接并增加来源注记，原文哈希与发布文本哈希分别记录。原JSON、CSV和样例数组保持原数值；压缩原始文件解压后字节与来源完全一致。
