# Paper3 研究思路演变

早期研究由 RF/VLC 信道估计误差和信息老化出发。两轮理论推导分别处理估计精度、服务可行性和资源代价，后续把保留的理论关系接到条件速率预测。CRLB 说明给定观测模型下的精度边界，候选动作的服务判断使用条件轨迹产生的周期速率分布。 [历史理论材料](https://github.com/qbbbh/paper3/tree/6ae23c04c091edee70e228cae39d72efb2af04a8/manuscripts/paper3_20261111/theory_history)

业务定义逐步确定为 100 ms 内固定的有向完整配对，每车承担发送或接收角色。功率口径统一为发送支路电输入，先按链路计算 EE，再计算算术平均及最弱链路。平均 EE 为初始参考及后续效率预算提供口径，公平阶段提高预测最低链路 EE。 [作者复盘](manuscripts/paper3_20261111/project_notes/00_重要复盘_发射功率与算法反复修改_20261024.md) [现行模型](MODEL_SPEC.md)

50 车配对与模式相同的对照仍出现最低 EE 差距，促使进一步检查固定结构下的联合功率搜索。现行实现以原求解结果为保留解，用块初后验生成候选，再按完整条件预测检查服务、平均 EE 预算和最低 EE。对照、实际损失、失败和时间均保留在同周期证据中。 [改进前诊断与动机](evidence/paper3_20261110/baseline_records/pre_guard_cross_scale/README.md) [全量结果](evidence/paper3_20261110/DATA_CATALOG.md)

Paper5 延续共同问题，研究图信息表示、协同学习决策和有限决策时间。 [Paper5 当前方案](https://github.com/qbbbh/paper5-revision)

历史源码、图表、草稿和完整论证链可在[固定提交 6ae23c0](https://github.com/qbbbh/paper3/tree/6ae23c04c091edee70e228cae39d72efb2af04a8)追溯。
