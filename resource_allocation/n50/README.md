# 50车：预留开发区，当前模型待适配

**不能宣称当前10车平均逐链路EE＋连续功率＋1 Mbit/s＋机会筛选已经在50车MASAC中完成。**

- [data](data/README.md)：该规模的场景原始数据和全量CSV，已准备。
- [src](src/README.md)：共享学习源码的明确入口；MLP为既有开发路径，不是新确认配置。
- [historical_results](historical_results/README.md)：既有短跑/诊断JSON与CSV，保留差结果和回退信息；不是当前物理设定的正式结果。
- [configs](configs/README.md)：后续批准配置的预留位置，尚不添加虚构“最终配置”。

后续需逐项核对：奖励是否平均逐链路EE；每种模式分母和功率定义；连续功率头；1 Mbit/s约束与风险口径；proposal/executed标签；失败/回退；训练与独立评估划分；同规模基线与推理开销。当前没有新的长跑授权。
