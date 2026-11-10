# Paper3：研究设定、实现与证据库

先看 [START_HERE.md](START_HERE.md)，再看 [MODEL_SPEC.md](MODEL_SPEC.md)。仓库为用户批准的公开仓库 https://github.com/qbbbh/paper3 。

当前整理的是源码 `95960f5a…` 对应的开发快照。**10车已有连续功率＋1 Mbit/s约束及mean/chance对照；50/100车尚未按这套设定完成调整和验收。** 这不是“所有方法已完成”的发布。

|目录|内容与状态|
|---|---|
|[channel_estimation](channel_estimation/README.md)|信道、观测、估计、预测库、CRLB与估计曲线|
|[resource_allocation/n10](resource_allocation/n10/README.md)|10车数值优化源码、数据、四组100块结果、原始配置|
|[resource_allocation/n50](resource_allocation/n50/README.md)|50车数据、已有诊断代码和历史学习结果；当前模型待适配|
|[resource_allocation/n100](resource_allocation/n100/README.md)|100车数据、图模型代码入口及历史学习结果；当前模型待适配|
|[resource_allocation/shared](resource_allocation/shared/README.md)|共用运行入口、数据读取；不重复维护三套实现|
|[resource_allocation/learning_shared](resource_allocation/learning_shared/README.md)|50/100共享MASAC、编码器、特征、信用诊断源码；开发状态|
|[references](references/INDEX.md)|文献目录、既有BibTeX、参数来源、阅读笔记与核实边界|
|[research_history](research_history/README.md)|研究演变、已犯错误与更正、原始文档；不能覆盖当前规格|
|[reviews](reviews/README.md)|复核结果、代码路径映射、来源及校验记录|
|[plans](plans/RESEARCH_PROPOSALS.md)|待审研究建议，包括三图一表布局；不是执行指令|
|[tests](tests/README.md)|原始单测、独立审计器和固定夹具|
|[tools](tools/README.md)|数据还原、CSV导出、校验和启动工具；不是优化算法|

## 开始阅读与运行

Chat可直接读Markdown、源码与CSV。NPZ为机器计算原件；每种规模的 `data/README.md`、`data/csv/mobility_preview.csv` 和完整分段CSV供在线审查。

首次准备：`python tools/run.py prepare`。快照验证：`python tools/verify_snapshot.py`。单测：`python tools/run.py test -q`。详细步骤见 [RUNBOOK.md](RUNBOOK.md)。

为了保留历史源码哈希，源码按用途展示，准备工具在 `.runtime/frozen_95960f5a/` 还原原导入布局；**科学源码内容不变**。运行缓存不提交。较大的二进制数据在 `data_parts/` 无损分段保存，准备时验证并还原。CSV和摘要可直接阅读。

GitHub记录版本，不能强制助手遵守模型；正式执行仍应固定commit并审查源码、配置、数据。当前没有自动实验工作流，不会因下载仓库而启动仿真。
