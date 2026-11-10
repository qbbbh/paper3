# Paper3 当前实现、论文材料与审核证据

更新：2026-11-11。先读 [START_HERE](START_HERE.md) 与 [独立审核提纲](REVIEW_BRIEF.md)。当前数值版本为 **paper3_20261110**；旧 `95960f5a` 快照及历史结果保留，用途与当前版本分开说明。

Paper3 的10车A、10车B、50车B各1000个计划周期已经完成核查。新增功率协调主要改善50车：每周期最低链路EE均值37.045→48.470 Mbit/J，平均EE下降2.40%，全链路实际服务991→994/1000；10车基本保持。相对Zappone，50车最低EE均值仍低0.43%。实际损失和少数退化周期完整保留，见[全量证据入口](evidence/paper3_20261110/README.md)。

| 内容 | 入口 |
|---|---|
| 当前业务、物理模型、目标和指标 | [MODEL_SPEC](MODEL_SPEC.md) |
| 当前源码、原11月6日对照源码、固定配置与测试 | [releases/paper3_20261110](releases/paper3_20261110/README.md) |
| 完整逐周期/逐链路表、原比较方法、原始数组样例与审计 | [evidence/paper3_20261110](evidence/paper3_20261110/README.md) |
| 当前引言、系统模型、论证对应、写作经验与两轮历史理论 | [manuscripts/paper3_20261111](manuscripts/paper3_20261111/README.md) |
| Paper5修改版：共同模型、新研究方案及复核 | [Paper5修改版](Paper5修改版/README.md) |
| 具体完成状态、统计对象与未完成写作 | [PROJECT_STATUS](PROJECT_STATUS.md) |
| 安装、检查、运行与历史入口 | [RUNBOOK](RUNBOOK.md) |

准备当前环境：`python -m pip install -r requirements.txt`。只做小型验证：

```bash
python releases/paper3_20261110/tools/check_release.py --tests
python releases/paper3_20261110/tools/run_case.py --case 50B
```

第二条仅预览配置；实际仿真需显式 `--execute --out <新目录>`。当前方法使用显式 `--maxmin-refine-solver guarded`，省略该选项仍执行旧NM。环境安装/打开仓库不应自动启动1000周期实验。

仓库提供代码、固定输入、全部紧凑结果与预定样例。现存大体积逐周期恢复状态保留在本地，未装成普通Git文件；完整历史replay仍需要相应原文件。不同历史批次的恢复点保留数量不同，以来源清单为准。当前代码及数据可独立审阅，样例可检查速率/EE公式；不能把有限样例称为全部真实轨迹已上传。

Paper5修改版目前是研究方案和复核材料，尚无对应新训练代码或已完成训练结果。旧50/100车学习代码位于历史目录，不代表新Paper5已经实现。

原根入口文档保存于 [archive/docs_before_20261111](archive/docs_before_20261111/ARCHIVE_README.md)。旧 `tools/run.py`、`source_layout_map` 和 `95960f5a` 仍服务历史复现；根清单中的旧科学哈希不会被追认给新版本。
