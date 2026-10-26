# 10车：当前连续功率数值实现

当前源码快照 `95960f5a…`。调整过的10车开发线，不等于论文全部实验已验收。

- [src/allocation.py](src/allocation.py)：完整匹配候选、邻域搜索、mean/chance筛选、回退。
- [src/continuous_power.py](src/continuous_power.py)：连续功率主实现。必须结合调用路径核对，不能用文件名判算法。
- [src/dinkelbach_sca.py](src/dinkelbach_sca.py)：历史兼容命名的可选精修；不能直接称经典Dinkelbach或全局最优。
- [共用入口](../shared/entrypoints/run_allocation.py)：含10/50/100功能，存在学习路径不代表学习已按当前规格验收。
- [信道预测库](../../channel_estimation/src/forecast.py)：用因果后验条件轨迹计算预测服务。
- [data](data/README.md)：10车场景与可读CSV。
- [results](results/README.md)：四组100块原始结果；配置见各自summary.json。
- [configs](configs/README.md)：从原run提取的配置，不另造默认值。

阅读顺序：MODEL_SPEC → 连续功率 → 匹配候选/筛选 → 同一run配置和逐链路CSV → 独立审计。

调用原入口帮助：`python tools/run.py allocation --help`（仓库根目录）。实际运行命令必须沿用审查批准的配置，不能因为整理目录而自动长跑。
