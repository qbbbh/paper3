# Paper3 当前实现、论文材料与审核证据

先读 [START_HERE](START_HERE.md) 与 [独立审核提纲](REVIEW_BRIEF.md)。当前实现与三组正式结果对应 `paper3_20261110`。

Paper3 的10车A、10车B、50车B各1000个计划周期已经完成核查。新增功率协调主要改善50车：每周期最低链路EE均值37.045→48.470 Mbit/J，平均EE下降2.40%，全链路实际服务991→994/1000；10车基本保持。相对Zappone，50车最低EE均值仍低0.43%。实际损失和少数退化周期完整保留，见[全量证据入口](evidence/paper3_20261110/README.md)。

| 内容 | 入口 |
|---|---|
| 当前业务、物理模型、目标和指标 | [MODEL_SPEC](MODEL_SPEC.md) |
| 当前源码、固定配置与测试 | [releases/paper3_20261110](releases/paper3_20261110/README.md) |
| 经校验的交通与建筑输入 | [data](data/INPUT_MANIFEST.json) |
| 完整逐周期/逐链路表、原比较方法、原始数组样例与审计 | [evidence/paper3_20261110](evidence/paper3_20261110/README.md) |
| 当前引言、系统模型、论证对应与写作经验 | [论文材料](manuscripts/paper3_20261111/README.md) |
| 研究问题与设计的演变 | [研究思路演变](RESEARCH_EVOLUTION.md) |
| Paper5 研究规格与复核材料 | [独立仓库 paper5-revision](https://github.com/qbbbh/paper5-revision) |
| 具体完成状态、统计对象与未完成写作 | [PROJECT_STATUS](PROJECT_STATUS.md) |
| 安装、检查、运行与历史入口 | [RUNBOOK](RUNBOOK.md) |

准备当前环境：`python -m pip install -r requirements.txt`。只做小型验证：

```bash
python releases/paper3_20261110/tools/check_release.py --tests
python releases/paper3_20261110/tools/run_case.py --case 50B
```

第二条仅预览配置；实际仿真需显式 `--execute --out <新目录>`。当前方法使用显式 `--maxmin-refine-solver guarded`，省略该选项仍执行旧NM。环境安装/打开仓库不应自动启动1000周期实验。

仓库提供代码、固定输入、全部紧凑结果与预定样例。现存大体积逐周期恢复状态保留在本地，未装成普通Git文件；完整历史replay仍需要相应原文件。不同历史批次的恢复点保留数量不同，以来源清单为准。当前代码及数据可独立审阅，样例可检查速率/EE公式；不能把有限样例称为全部真实轨迹已上传。

需要逐字核对旧对照源码、历史推导或旧稿时，可查阅[上一版固定提交](https://github.com/qbbbh/paper3/tree/6ae23c04c091edee70e228cae39d72efb2af04a8)。
